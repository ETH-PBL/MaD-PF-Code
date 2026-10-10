import hashlib
import inspect
import os
from dataclasses import dataclass
from functools import wraps
from datetime import datetime
from typing import Any, Callable, Iterable, Optional
import getpass
import sys
import base64

import torch
import wandb
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import deprecated
from pathlib import Path

from .module import ConfigRegistry, ModuleConfig


def get_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device('cuda')

  if hasattr(torch, 'mps') and torch.mps.is_available():
    return torch.device('mps')

  return torch.device('cpu')


def sample_frame(frame: Tensor, samples: int) -> Tensor:
  return frame[torch.randperm(frame.shape[0])[:samples]]


@deprecated('migrate to udq poses')
def get_tf(a: Tensor, t: Tensor) -> Tensor:
  P = torch.zeros(*torch.broadcast_shapes(a.shape, t.shape[:-1]), 4, 4, device=a.device, dtype=a.dtype)
  c = torch.cos(a)
  s = torch.sin(a)
  P[..., 0, 0] = c
  P[..., 0, 1] = -s
  P[..., 1, 0] = s
  P[..., 1, 1] = c
  P[..., 2, 2] = 1
  P[..., 0, 3] = t[..., 0]
  P[..., 1, 3] = t[..., 1]
  P[..., 3, 3] = 1
  return P


@deprecated('migrate to udq poses')
def apply_tf(points: Tensor, tf: Tensor) -> Tensor:
  return torch.cat([
    points[..., :3] @ tf[..., :3, :3].transpose(-1, -2) + tf[..., :3, 3].unsqueeze(-2),
    points[..., 3:].expand(*torch.broadcast_shapes(points.shape[:-2], tf.shape[:-2]), -1, -1)
  ], dim=-1)


@deprecated('migrate to udq poses')
def tf_log(tf: Tensor) -> Tensor:
  # pseudo log for SE(2) -> se(2)
  return torch.cat([tf[..., :2, 0], tf[..., :2, 3]], dim=-1)


@deprecated('migrate to udq poses')
def tf_exp(tf: Tensor) -> Tensor:
  # pseudo exp for se(2) -> SE(2)
  return get_tf(torch.atan2(tf[..., 1], tf[..., 0]), tf[..., 2:])


def all_to(*data, dtype: torch.dtype = None, device: torch.device = None) -> Iterable:
  yield from (x.to(dtype=dtype, device=device) if isinstance(x, Tensor) else x for x in data)


def collate_fn(batch):
  return tuple(torch.cat(items) for items in zip(*batch))


def disk_cache(*key_args: str):
  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      bound_args = inspect.signature(func).bind(*args, **kwargs)
      bound_args.apply_defaults()

      storage_path = os.path.join(
        'cache',
        os.path.splitext(os.path.basename(inspect.getfile(func)))[0],
        func.__name__,
        hashlib.md5(repr([bound_args.arguments[k] for k in key_args]).encode()).hexdigest() + '.pt'
      )

      if not os.path.exists(storage_path):
        out = func(*args, **kwargs)
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        torch.save(out, storage_path)
      else:
        out = torch.load(storage_path)

      return out
    return wrapper
  return decorator


def parse_dataset_input(data_string: str) -> list[Path]:
  """
  Parses a comma-separated string into a list of paths.
  The first element is the root, and all subsequent elements 
  are subdirectories of that root.
  """

  parts = data_string.split(',')
  root = parts[0]
  return [Path(os.path.join(root, sub)) for sub in parts[1:]]


def train(
    model: nn.Module,
    opt: Optimizer,
    loss_fn: Callable[[nn.Module, Any, Any], Tensor],
    epochs: int,
    tdl: DataLoader,
    vdl: DataLoader = None,
    device: torch.device = None,
    loss_args=None,
    acc_steps: int = 1,
    save_tag: Optional[str] = None,
    save_epochs: bool = True,
    use_amp: bool = True,
    patience: int = 0,
    min_delta: float = 0.0,
    config: Optional[ModuleConfig] = None,
) -> tuple[list[float], list[float]]:
  """Early stop on val loss when patience>0 and vdl is given; saved model.pt is the best epoch.

  Checkpoints are config bundles ({'state_dict', 'config'}); `config` is required
  whenever `save_tag` is set so the model can be rebuilt via ModuleConfig.load_module.
  """
  if save_tag and config is None:
    raise ValueError('train(save_tag=...) requires a config to persist checkpoint bundles')

  config_payload = ConfigRegistry.encode(config) if config is not None else None

  def save_ckpt(p):
    torch.save({'state_dict': model.state_dict(), 'config': config_payload}, p)

  if save_tag:
    storage_path = os.path.join('models', save_tag)
    os.makedirs(storage_path, exist_ok=True)
    path = os.path.join(storage_path, 'init.pt')
    save_ckpt(path)
    wandb_save_artifact(path, f"{save_tag}-init", "model")
  else:
    storage_path = None

  training_loss = []
  validation_loss = []

  model.to(device=device)

  scaler = torch.amp.grad_scaler.GradScaler(device.type, enabled=(use_amp and device.type == 'cuda'))

  best_metric = float('inf')
  stale = 0
  for e in range(epochs):
    total = 0
    model.train()
    opt.zero_grad()
    for i, batch in enumerate(tqdm(tdl, desc='training')):
      batch = all_to(*batch, device=device)
      with torch.amp.autocast(device.type, enabled=(use_amp and device.type == 'cuda')):
        loss = loss_fn(model, batch, loss_args) / acc_steps
      scaler.scale(loss).backward()
      total += loss.item()

      if (i + 1) % acc_steps == 0 or i + 1 == len(tdl):
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()

    training_loss.append(total / len(tdl))
    # print(f'training loss: {training_loss[-1]:.6f} ({e + 1} / {epochs})')
    # print(plot([training_loss], {'height': 6, 'format': '{:10.6f} ', 'colors': [red]}))

    log_dict = {"train/loss": training_loss[-1]}

    if vdl:
      total = 0
      model.eval()
      for batch in tqdm(vdl, desc='evaluating'):
        batch = all_to(*batch, device=device)
        with torch.amp.autocast(device.type, enabled=(use_amp and device.type == 'cuda')):
          total += loss_fn(model, batch, loss_args).item() / acc_steps

      validation_loss.append(total / len(vdl))
      # print(f'validation loss: {validation_loss[-1]:.6f} ({e + 1} / {epochs})')
      # print(plot([validation_loss], {'height': 6, 'format': '{:10.6f} ', 'colors': [blue]}))
      log_dict["val/loss"] = validation_loss[-1]

    if wandb.run is not None:
      wandb.log(log_dict, step=e + 1)

    if storage_path:
      if save_epochs:
        save_ckpt(os.path.join(storage_path, f'epoch-{e:03d}.pt'))
      torch.save({
        'training': training_loss,
        'validation': validation_loss
      }, os.path.join(storage_path, 'loss.pt'))

    if patience > 0 and vdl:
      if best_metric - validation_loss[-1] > min_delta:
        best_metric = validation_loss[-1]
        stale = 0
        if storage_path:
          save_ckpt(os.path.join(storage_path, 'model.pt'))
      else:
        stale += 1
        if stale >= patience:
          print(f'early stop at epoch {e + 1} (no improvement >{min_delta} for {patience} epochs, best={best_metric:.6f})')
          break

  if storage_path and not (patience > 0 and vdl):
    path = os.path.join(storage_path, 'model.pt')
    save_ckpt(path)
    wandb_save_artifact(path, f"{save_tag}-final", "model")
    if wandb.run is not None:
      import json
      with open(os.path.join(storage_path, 'wandb_run.json'), 'w') as f:
        json.dump({
          'entity': wandb.run.entity,
          'project': wandb.run.project,
          'group': wandb.run.group,
          'run_id': wandb.run.id,
        }, f)

  return training_loss, validation_loss


def wandb_save_artifact(path: str | Path, name: str, type: str, metadata: dict | None = None):
  if wandb.run is not None:
    artifact = wandb.Artifact(name=name, type=type, metadata=metadata)
    artifact.add_file(str(path))
    wandb.run.log_artifact(artifact)



@dataclass
class WBConfig:
  enable: bool = False
  """enable WandB logging"""

  entity: str = 'ethz-pbl'
  """WandB entity (team or user)"""

  project: str = 'diff-pf'
  """WandB project name"""

  name: str | None = None
  """WandB run name, defaults to eval_UID"""

  group: str | None = None
  """WandB group name, defaults to save_tag"""


def wandb_init_2(args: WBConfig | None, info: dict | str | None = None):
  if args is None or not args.enable:
    return None

  if isinstance(info, dict):
    info = {k: v for k, v in info.items() if not isinstance(v, WBConfig)}

  wandb.init(
    entity=getattr(args, 'wandb_entity', None) or None,
    project=args.project,
    name=args.name or datetime.now().strftime('%Y%m%d_%H%M%S'),
    group=args.group,
    config=info
  )
  return wandb.run



def unlock_wandb():
  """
  Unlocks W&B by reading a base64-encoded 'password:api_key' string from 'wandb_secret'.
  To generate the 'wandb_secret' file, you can use:
    echo "your_password:your_api_key" | python -c "import base64, sys; print(base64.b64encode(sys.stdin.read().strip().encode()).decode())" > wandb_secret
  """
  if os.environ.get("WANDB_API_KEY"): # Already Set
    return

  if not os.path.exists("wandb_secret"):
    return

  with open("wandb_secret", "r") as f:
    scrambled_data = f.read().strip()

  user_input = getpass.getpass("Enter W&B password: ")
  decoded = base64.b64decode(scrambled_data.encode("utf-8")).decode("utf-8")
  stored_password, api_key = decoded.split(":", 1)
  if user_input == stored_password:
      os.environ["WANDB_API_KEY"] = api_key
      print("Access granted.")
  else:
      print("Incorrect password.")
      sys.exit(1)

unlock_wandb()