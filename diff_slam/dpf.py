from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional

import torch
import tyro
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from diff_slam.resample import ResamplingStrategy, ResampleConfig, ContractResampleConfig, RESAMPLE_CONFIGS
from .bag_parser import LocalisationDataset
from .module import ConfigRegistry, ModuleConfig
from .obs import obs_mode_label, ObservationModel
from .dynamics import (
  build_dynamics_config,
  DynamicsConfig,
  DynamicsModel,
  mode_from_dynamics,
)
from .pose import udq_transform_points_fast, udq_mean, udq_log_gaussian, udq_mul, udq_conj, udq_distance
import wandb
from .util import get_device, sample_frame, train, collate_fn, parse_dataset_input, wandb_init_2, WBConfig

class SequenceDataset(Dataset):

  def __init__(self, ds, step_count: int, downsample: int = 0):
    self.ds = ds
    self.step_count = step_count
    self.downsample = downsample

  def __len__(self):
    return len(self.ds) - (self.step_count - 1)

  def __getitem__(self, i):
    items = [self.ds[i + j] for j in range(self.step_count)]
    world = items[0][0]
    scans = torch.stack([sample_frame(item[1], self.downsample) if self.downsample > 0 else item[1] for item in items])
    poses = torch.stack([item[2] for item in items])
    odometry = torch.stack([item[3] for item in items])
    return world, scans, poses, odometry


class DPFv2(nn.Module):

  def __init__(self,
               dynamics: DynamicsModel,
               observer: ObservationModel,
               resample: ResamplingStrategy,
               particle_count: int,
               temp: float,
               init_noise: tuple[float, float] = (0.1, 0.05),
               odom_noise: tuple[float, float] = (0.05, 0.025)):
    super().__init__()
    self.dynamics = dynamics
    self.observer = observer
    # Hyperparameters are owned by DPFConfig (the persistence source of truth);
    # here they are plain runtime attributes, not buffers.
    self.resample = resample
    self.temp = temp
    self.particle_count = particle_count
    self.init_noise = init_noise
    self.odom_noise = odom_noise
    self.p = None

  def _get_w(self, p, z, world):
    zp = udq_transform_points_fast(p, z.expand(p.shape[0], -1, -1))
    return self.observer(zp, world)

  def forward(self, scans: torch.Tensor, x0: torch.Tensor, p0: torch.Tensor, inputs: torch.Tensor, world: torch.Tensor | None = None) -> torch.Tensor:
    x_out = x0.new_zeros(scans.shape[0] - 1, x0.shape[-1])

    p = p0
    resample_fn = self.resample

    for t, (z, u) in enumerate(zip(scans[1:], inputs)):
      # prediction step
      p = self.dynamics(p, u)

      # update step
      w = checkpoint(self._get_w, p, z, world, use_reentrant=False)
      w = torch.softmax(w / self.temp, dim=0)

      x_out[t] = udq_mean(p, w)
      p = resample_fn(p, w)
      
      if t == 0:
        self.p = p

    return x_out


@dataclass
@ConfigRegistry.register
class DPFConfig(ModuleConfig):
  dynamics: DynamicsConfig
  observer: ModuleConfig
  resample: ResampleConfig = field(default_factory=ContractResampleConfig)
  particle_count: int = 192
  temp: float = 1.0
  init_noise: tuple[float, float] = (1e-9, 1e-9)
  odom_noise: tuple[float, float] = (1e-9, 1e-9)

  steps: int = 6
  """sequence length for DPF training (data-loading param; unused by DPFv2.forward)"""

  downsample: int = 8000
  """point-cloud downsample target (data-loading param)"""

  def build_model(self) -> 'DPFv2':
    return DPFv2(
      dynamics=self.dynamics.build_model(),
      observer=self.observer.build_model(),
      resample=self.resample.get_strategy(),
      particle_count=self.particle_count,
      temp=self.temp,
      init_noise=self.init_noise,
      odom_noise=self.odom_noise,
    )


def init_se2_like_particles(count, std_t=1e-3, std_r=1e-3):
  floor = 1e-8
  return udq_log_gaussian(
    cov_t=torch.diag(torch.tensor([std_t ** 2, std_t ** 2, floor])),
    cov_r=torch.diag(torch.tensor([floor, floor, std_r ** 2])),
    size=(count,)
  )


def loss_fn(dpf, batch, args) -> torch.Tensor:
  world, scans, poses, odometry = batch

  x0 = poses[0]
  p0 = udq_mul(init_se2_like_particles(dpf.particle_count, dpf.init_noise[0], dpf.init_noise[1]).to(world.device), x0)
  u = udq_mul(udq_conj(odometry[:-1]), odometry[1:])

  if dpf.training:
    # data augmentation: add noise to odometry inputs
    floor = 1e-8
    std_t = dpf.odom_noise[0]
    std_r = dpf.odom_noise[1]
    du = udq_log_gaussian(
      torch.diag(torch.tensor([std_t ** 2, std_t ** 2, floor])),
      torch.diag(torch.tensor([floor, floor, std_r ** 2])),
      size=u.shape[:-1]
    ).to(u.device)
    u = udq_mul(u, du)

  x_out = dpf(scans, x0, p0, u, world)

  return torch.sum(udq_distance(x_out, poses[1:]))


@dataclass
class DPFTrainingParams:
  obs_model: str = 'models/obs/model.pt'
  """Pretrained observation-model checkpoint (a config bundle)."""

  train_ds: str = 'data/etf_floorF,d1,d2,d3,d6,d7,d8,d9,d10,f1,f3,f4,f5,f6,f8'
  """Training datasets. Format as 'data/DATASET_NAME,BAG1,BAG2,...'"""

  val_ds: str = 'data/etf_floorF,d4,d5,f2,f7'
  """Validation datasets, same format as train_ds."""

  save_tag: str = ''
  """Optional suffix on the models/dpf-<obs>-<motion>[-<tag>] output directory."""

  epochs: int = 1
  optimizer: Literal['adam', 'sgd'] = 'adam'
  steps: int = 6
  """Sequence length (frames) per DPF training sample."""

  downsample: int = 8000
  """Point-cloud downsample target."""

  particles: int = 192
  temp: float = 1.0
  """Softmax temperature for particle weighting."""

  init_noise: tuple[float, float] = (1e-9, 1e-9)
  """Initial particle jitter (trans, rot)."""

  odom_noise: tuple[float, float] = (1e-9, 1e-9)
  """Odometry augmentation noise during training (trans, rot)."""

  proc_noise: tuple[float, float] = (0.07, 0.05)
  """Process noise for a fresh motion_arch model (trans, rot)."""

  motion: Optional[str] = None
  """Pretrained motion-model checkpoint (mutually exclusive with motion_arch)."""

  motion_arch: Optional[Literal['odometry-se3', 'realnvp-se3', 'ensemble-se3', 'mlp-se3']] = None
  """Fresh motion-model architecture (mutually exclusive with motion)."""

  resample: Literal['contract', 'gumbel', 'ot'] = 'contract'


@dataclass
class TrainDPF:
  """Train a differentiable particle filter end to end."""

  train: DPFTrainingParams = field(default_factory=DPFTrainingParams)
  wandb: WBConfig = field(default_factory=WBConfig)


def train_dpf(cmd: TrainDPF):
  device = get_device()
  p = cmd.train

  train_ds = ConcatDataset([SequenceDataset(LocalisationDataset(path=d), p.steps, p.downsample) for d in parse_dataset_input(p.train_ds)])
  val_ds = ConcatDataset([SequenceDataset(LocalisationDataset(path=d), p.steps, p.downsample) for d in parse_dataset_input(p.val_ds)])
  tdl = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn)
  vdl = DataLoader(val_ds, shuffle=True, collate_fn=collate_fn)

  # Pretrained observation model: load its config + weights from the bundle.
  obs_cfg, obs_model = ModuleConfig.load_module(Path(p.obs_model))
  mode = obs_mode_label(obs_cfg)

  # Motion model: pretrained checkpoint or a fresh architecture.
  if (p.motion is None) == (p.motion_arch is None):
    raise ValueError("Specify exactly one of motion or motion_arch")
  if p.motion is not None:
    motion_cfg, motion_model = DynamicsConfig.load_module(Path(p.motion))
  else:
    assert p.motion_arch is not None
    motion_cfg = build_dynamics_config(p.motion_arch, p.proc_noise[0], p.proc_noise[1])
    motion_model = motion_cfg.build_model()
  motion_mode = mode_from_dynamics(motion_model)

  resample_cfg = RESAMPLE_CONFIGS[p.resample]()

  config = DPFConfig(
    dynamics=motion_cfg,
    observer=obs_cfg,
    resample=resample_cfg,
    particle_count=p.particles,
    temp=p.temp,
    init_noise=p.init_noise,
    odom_noise=p.odom_noise,
    steps=p.steps,
    downsample=p.downsample,
  )

  save_tag = f'dpf-{mode}-{motion_mode}' + (f'-{p.save_tag}' if p.save_tag else '')
  model = DPFv2(
    dynamics=motion_model,
    observer=obs_model,
    resample=resample_cfg.get_strategy(),
    particle_count=p.particles,
    temp=p.temp,
    init_noise=p.init_noise,
    odom_noise=p.odom_noise,
  ).to(device)

  optimizer = (
    torch.optim.Adam(model.parameters(), lr=1e-5) if p.optimizer == 'adam'
    else torch.optim.SGD(model.parameters(), lr=1e-5)
  )

  wandb_init_2(cmd.wandb, asdict(cmd))

  result = train(model, optimizer, loss_fn, p.epochs, tdl, vdl, device, save_tag=save_tag, config=config)
  if wandb.run is not None:
    wandb.finish()
  return result


if __name__ == '__main__':
  train_dpf(tyro.cli(TrainDPF))
