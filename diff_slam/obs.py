from abc import abstractmethod, ABC
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import tyro
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.nn import MSELoss
from torch.utils.data import ConcatDataset, DataLoader

from diff_slam.module import ConfigRegistry, ModuleConfig
from .bag_parser import LocalisationDataset, make_world
from .util import sample_frame, get_device, train, collate_fn, parse_dataset_input, wandb_save_artifact, wandb_init_2, WBConfig
from .pose import udq_mul, udq_transform_points, udq_from_qt


class ObservationModel(nn.Module, ABC):

  @abstractmethod
  def forward(self, points: torch.Tensor, world: torch.Tensor | None) -> torch.Tensor: ...


class MultiSDF(nn.Module):

  def __init__(self, grids: tuple[int, ...], e_dim: int, h_dim: int):
    super().__init__()
    self.register_buffer('offset', torch.zeros(3))
    self.register_buffer('scale', torch.ones(3))

    self.grids = nn.ParameterList(
      nn.Parameter(torch.randn(1, e_dim, g, g, g) * 0.01) for g in grids
    )
    self.s0 = nn.Sequential(
      nn.Linear(len(grids) * e_dim, h_dim),
      nn.ReLU(inplace=True),
      nn.Linear(h_dim, h_dim),
      nn.ReLU(inplace=True),
      nn.Linear(h_dim, 1)
    )

  def forward(self, points: torch.Tensor) -> torch.Tensor:
    pos = points[..., :3].view(-1, 3)
    pos = (pos - self.offset) / self.scale
    pos = pos.view(1, -1, 1, 1, 3)

    e = []
    for grid in self.grids:
      x = F.grid_sample(grid, pos, mode='bilinear', padding_mode='border', align_corners=True)
      e.append(x.view(grid.shape[1], -1))

    return self.s0(torch.cat(e, dim=0).T).view(*points.shape[:-1])


@dataclass
@ConfigRegistry.register
class SDFModuleConfig(ModuleConfig):
  grids: tuple[int, ...] = (16, 32, 64, 128)
  """spatial embedding grid resolutions"""

  e_dim: int = 32
  """embedding vector size per grid level"""

  h_dim: int = 64
  """decoder hidden layer size"""

  def build_model(self) -> MultiSDF:
    return MultiSDF(grids=self.grids, e_dim=self.e_dim, h_dim=self.h_dim)


class SDFObservationModel(ObservationModel):

  def __init__(self, sdf: MultiSDF, sigma: float, delta: float):
    super().__init__()
    self.sdf = sdf
    self.register_buffer('sigma', torch.tensor(sigma))
    self.register_buffer('delta', torch.tensor(delta))

  def forward(self, points: torch.Tensor, world: torch.Tensor | None = None) -> torch.Tensor:
    dist = self.sdf(points)
    err = F.huber_loss(dist, dist.new_zeros(1).expand_as(dist), reduction='none', delta=self.delta.item())
    return torch.exp(-self.sigma * err.mean(dim=-1)).unsqueeze(-1)


@dataclass
@ConfigRegistry.register
class SDFOBSModuleConfig(ModuleConfig):
  sdf: SDFModuleConfig = field(default_factory=SDFModuleConfig)
  sigma: float = 80.0
  """exponential output score decay"""

  delta: float = 0.2
  """huber loss threshold"""

  def build_model(self) -> SDFObservationModel:
    return SDFObservationModel(
      sdf=self.sdf.build_model(),
      sigma=self.sigma,
      delta=self.delta
    )


@dataclass
class SDFTrainingParams:
  train_ds: str
  """Format as 'data/DATASET_NAME,BAG1,BAG2,...'"""

  save_tag: str = 'sdf'
  epochs: int = 500
  batch_size: int = 8192
  eikonal_loss_weight: float = 0.1
  noise_scale: float = 0.3
  show_map: bool = False


def train_sdf(model: MultiSDF, params: SDFTrainingParams, world: torch.Tensor, device: torch.device):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

  world_min = world[:, :3].amin(dim=0)
  world_max = world[:, :3].amax(dim=0)
  model.offset = (world_max + world_min) / 2
  model.scale = (world_max - world_min) / 2
  model.train().to(device=device)

  eps: float = 1e-5
  dx = torch.tensor([eps, 0.0, 0.0], device=device)
  dy = torch.tensor([0.0, eps, 0.0], device=device)
  dz = torch.tensor([0.0, 0.0, eps], device=device)

  progress = tqdm(range(params.epochs))
  for e in progress:
    optimizer.zero_grad()

    i = torch.randint(0, world.shape[0], (params.batch_size,), device=world.device)
    p0 = world[i, :3].to(device=device)

    noise = torch.randn_like(p0) * params.noise_scale
    pn = p0 + noise

    # reference points: d = 0
    o0 = model(p0)
    reference_loss = F.l1_loss(o0, o0.new_zeros(1).expand_as(o0))

    # offset points: |d| >= 0
    on = model(pn)
    offset_loss = F.l1_loss(on, torch.norm(noise, dim=-1))

    # eikonal soft constraint: |\nabla f| = 1
    norm = torch.stack([
      model(pn + dx) - model(pn - dx),
      model(pn + dy) - model(pn - dy),
      model(pn + dz) - model(pn - dz),
    ], dim=-1).norm(dim=-1) / (2 * eps)
    eikonal_loss = F.mse_loss(norm, norm.new_ones(1).expand_as(norm))

    loss = reference_loss + offset_loss + params.eikonal_loss_weight * eikonal_loss

    loss.backward()
    optimizer.step()

    if (e + 1) % 25 == 0:
      if wandb.run is not None:
        wandb.log({'train/loss': loss.item()}, step=e + 1)
      progress.set_postfix(loss=f'{loss.item():.4f}')

  return model


@dataclass
class OBS_SDF:
  train: SDFTrainingParams
  model: SDFOBSModuleConfig
  wandb: WBConfig


def main_sdf(cmd: OBS_SDF):
  world = make_world(parse_dataset_input(cmd.train.train_ds))

  if cmd.train.show_map:
    from .visualize import show_world_np
    show_world_np(sample_frame(world, 5000))

  device = get_device()

  if device.type == 'mps':
    device = torch.device('cpu')  # grid_sample not implemented yet :/

  wandb_init_2(cmd.wandb, asdict(cmd))

  obs_model = cmd.model.build_model()
  train_sdf(obs_model.sdf, cmd.train, world, device)

  out_path = Path(f'models/{cmd.train.save_tag}/model.pt')
  cmd.model.save_module(obs_model, out_path)
  wandb_save_artifact(out_path, f'{cmd.train.save_tag}-final', 'model')

  if wandb.run is not None:
    wandb.finish()


class AlignmentDataset(LocalisationDataset):

  def __init__(self, path: Path, downsample_scan: int = 0, downsample_world: int = 0, sigma_scale: float = 1.0, step: int = 1, map_device: torch.device = None, lidar_topic: str = '/livox/lidar', odometry_topic: str = '/odometry/filtered', noise_scale: float = 1.0):
    super().__init__(path, lidar_topic=lidar_topic, odometry_topic=odometry_topic)
    self.downsample = downsample_scan
    self.step = step
    self.sigma_scale = sigma_scale
    self.noise_scale = noise_scale
    self.map_device = torch.device('cpu') if map_device is None else map_device

    if downsample_world > 0:
      self.world = sample_frame(self.world, downsample_world)

  def __len__(self):
    return super().__len__() // self.step

  @staticmethod
  def get_positioned(world, scan, pose):
    s = udq_transform_points(pose, scan)
    m = s.mean(dim=-2, keepdim=True)
    m[..., 2:] = 0
    return world - m, s - m

  @staticmethod
  def gaussian(x, s):
    return torch.exp(-(x ** 2) / (2 * s ** 2))

  def get_perturbed(self, scan, pose, offset):
    perturbed = udq_mul(pose, offset)
    return self.get_positioned(self.world.to(self.map_device), scan, perturbed.to(self.map_device))

  def __getitem__(self, idx):
    t_noise = torch.rand(5, 2) * torch.tensor([0.0, 0.5, 2.0, 2.0, 0.0]).unsqueeze(1) * self.noise_scale
    r_noise = torch.rand(5) * torch.tensor([0.0, 0.1, 0.3, 0.0, 0.3]) * self.noise_scale

    scan = self.scans[idx * self.step].clone().to(self.map_device)
    if self.downsample > 0:
      scan = sample_frame(scan, self.downsample)

    q_noise = torch.stack([
      torch.cos(r_noise / 2),
      torch.zeros_like(r_noise),
      torch.zeros_like(r_noise),
      torch.sin(r_noise / 2)
    ], dim=-1)
    t_noise_3d = torch.cat([t_noise, torch.zeros_like(r_noise).unsqueeze(-1)], dim=-1)
    offset = udq_from_qt(q_noise, t_noise_3d)

    W, S = self.get_perturbed(scan, self.poses[idx * self.step], offset)
    Y = self.gaussian(torch.norm(t_noise, dim=-1), 0.5 * self.sigma_scale) * self.gaussian(r_noise, 0.3 * self.sigma_scale)

    yaw = torch.rand(1, device=self.map_device) * 2 * torch.pi
    tf_q = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], dim=-1)
    tf_t = torch.zeros((1, 3), device=self.map_device)
    tf = udq_from_qt(tf_q, tf_t)

    W = udq_transform_points(tf, W)
    S = udq_transform_points(tf, S)

    return W, S, Y.unsqueeze(-1).to(self.map_device)


def rasterize_pool(xy: torch.Tensor, data: torch.Tensor, grid_size: torch.Tensor, fill_value: float, reduce: Literal['amax', 'amin', 'sum', 'mean', 'prod']) -> torch.Tensor:
  W, H = grid_size
  *batch_dims, N, _ = xy.shape
  M = data.shape[-1]
  B = int(torch.tensor(batch_dims).prod().item()) if batch_dims else 1

  xy_flat = xy.reshape(B, N, 2)
  data_flat = data.reshape(B, N, M)

  xi, yi = xy_flat[..., 0].floor().int(), xy_flat[..., 1].floor().int()
  mask = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
  i = torch.where(mask, xi * H + yi, W * H)

  out = torch.full((B, W * H + 1, M), fill_value, dtype=data.dtype, device=data.device)
  out.scatter_reduce_(dim=1, index=i.unsqueeze(-1).expand(-1, -1, M), src=data_flat, reduce=reduce, include_self=True)

  return out[:, :-1, :].reshape(*batch_dims, W, H, M) if batch_dims else out[:, :-1, :].reshape(W, H, M)


class FixedPillarEncoder(nn.Module):

  def __init__(self, target_size: torch.Tensor, grid_size: float, z_scale: float):
    super().__init__()
    self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.float32))
    self.register_buffer('z_scale', torch.tensor(z_scale, dtype=torch.float32))
    self.register_buffer('target_size', target_size[:2].to(dtype=torch.int32))

  def encode(self, xy: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    counts = rasterize_pool(xy, torch.ones_like(points[..., [2]]), self.target_size.cpu().numpy(), reduce='sum', fill_value=0)

    zl = points[..., [2]]
    zl[zl > 0.5] = 0

    zu = points[..., [2]]
    zu[zu < 0.5] = 1

    # 0: occupancy
    # 1: lower profile
    # 2: upper profile
    # 3: mean intensity
    # 4: log density
    # 5: x mean
    # 6: y mean

    return torch.cat([
      counts.clip(0, 1),
      # TODO: don't use NumPy?
      rasterize_pool(xy, zl, self.target_size.cpu().numpy(), 0, 'amax') * 2,
      rasterize_pool(xy, zu, self.target_size.cpu().numpy(), 1, 'amin') * 2 - 1,
      rasterize_pool(xy, points[..., [3]], self.target_size.cpu().numpy(), 0, 'mean'),
      torch.log1p(counts) / 10,
      rasterize_pool(xy, points[..., :2], self.target_size.cpu().numpy(), 0, 'mean'),
    ], dim=-1).transpose(-1, -3)

  def forward(self, p):
    p = p + torch.cat([
      self.target_size.cpu() * self.grid_size.cpu() / 2,
      torch.zeros(2)
    ], dim=-1).to(device=p[0].device)

    xy = p[..., :2] / self.grid_size

    v0 = torch.cat([
      torch.floor(xy) * self.grid_size + self.grid_size / 2,
      torch.zeros_like(p[..., 2:])
    ], dim=-1)
    vs = torch.tensor([
      2 / self.grid_size,
      2 / self.grid_size,
      self.z_scale,
      1 / 255
    ], device=p.device)

    return self.encode(xy, (p - v0) * vs)


class StructuralComparator(ObservationModel):

  def __init__(self, encoder: FixedPillarEncoder, pool: bool):
    super().__init__()
    self.pool = pool
    self.encoder = encoder

    self.s0 = nn.Sequential(
      self._reduction_block(14, 32),
      nn.MaxPool2d(kernel_size=2) if self.pool else nn.Identity(),
      self._reduction_block(32, 64),
      self._reduction_block(64, 128),
      nn.AdaptiveAvgPool2d(1),
      nn.Flatten(),
      nn.Linear(128, 1),
      nn.Sigmoid()
    )

  @staticmethod
  def _reduction_block(in_channels: int, out_channels: int):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.GroupNorm(out_channels // 8, out_channels),
      nn.ReLU(),
    )

  def forward(self, scan: torch.Tensor, world: torch.Tensor | None):
    world_bev = self.encoder(world)
    scan_bev = self.encoder(scan)

    return self.s0(torch.cat([
      world_bev.expand(scan_bev.shape),
      scan_bev
    ], dim=1))


@dataclass
@ConfigRegistry.register
class PPObserverConfig(ModuleConfig):
  pool: bool = True
  """enable pooling layer"""

  grid_size: float = 0.15
  """grid cell side length"""

  grid_z_scale: float = 0.154
  """vertical point cloud scaling factor"""

  target_size: tuple[int, int] = (100, 100)
  """structural comparator grid window size"""

  def build_model(self) -> StructuralComparator:
    return StructuralComparator(
      encoder=FixedPillarEncoder(
        target_size=torch.tensor(self.target_size, dtype=torch.int32),
        grid_size=self.grid_size,
        z_scale=self.grid_z_scale
      ),
      pool=self.pool
    )


def obs_mode_label(cfg: ModuleConfig) -> str:
  """Short label for an observation-model config (filenames / metrics 'mode')."""
  if isinstance(cfg, SDFOBSModuleConfig):
    return 'sdf'
  if isinstance(cfg, PPObserverConfig):
    return 'ppp' if cfg.pool else 'ppe'
  return type(cfg).__name__


def loss_fn(obs, batch, args) -> Tensor:
  M, z, y = batch
  loss, = args
  return loss(obs(z, M), y)


@dataclass
class PPTrainingParams:
  train_ds: str
  val_ds: str
  save_tag: str = 'pp'
  downsample_scan: int = 0
  downsample_world: int = 0
  sigma_scale: float = 0.3
  noise_scale: float = 1.0
  batch_size: int = 6
  epochs: int = 15
  patience: int = 5


@dataclass
class OBS_PP:
  train: PPTrainingParams
  model: PPObserverConfig
  wandb: WBConfig


def main_pp(cmd: OBS_PP):
  device = get_device()

  ds_kwargs = {
    'downsample_scan': cmd.train.downsample_scan,
    'downsample_world': cmd.train.downsample_world,
    'sigma_scale': cmd.train.sigma_scale,
    'map_device': device,
    'noise_scale': cmd.train.noise_scale,
  }

  dl_kwargs = {
    'batch_size': cmd.train.batch_size,
    'shuffle': True,
    'collate_fn': collate_fn
  }

  train_ds = ConcatDataset([AlignmentDataset(p, **ds_kwargs) for p in parse_dataset_input(cmd.train.train_ds)])
  val_ds = ConcatDataset([AlignmentDataset(p, step=10, **ds_kwargs) for p in parse_dataset_input(cmd.train.val_ds)])

  tdl = DataLoader(train_ds, **dl_kwargs)
  vdl = DataLoader(val_ds, **dl_kwargs)

  epochs = cmd.train.epochs

  wandb_init_2(cmd.wandb, asdict(cmd))

  obs_model = cmd.model.build_model()
  optimizer = torch.optim.Adam(obs_model.parameters(), lr=1e-3)
  train(obs_model, optimizer, loss_fn, epochs, tdl, vdl, device, (MSELoss(),), save_tag=cmd.train.save_tag, patience=cmd.train.patience, config=cmd.model)

  if wandb.run is not None:
    wandb.finish()


if __name__ == '__main__':
  _cmd = tyro.cli(OBS_SDF | OBS_PP)
  if isinstance(_cmd, OBS_SDF):
    main_sdf(_cmd)
  if isinstance(_cmd, OBS_PP):
    main_pp(_cmd)
