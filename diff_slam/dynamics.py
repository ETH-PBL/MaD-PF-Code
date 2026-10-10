import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional, Protocol, cast

import torch
import tyro
import wandb
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from .bag_parser import LocalisationDataset
from .module import ConfigRegistry, ModuleConfig
from .pose import q_conj, q_mul, udq_conj, udq_from_qt, udq_log_gaussian, udq_mul
from .util import (
  WBConfig,
  collate_fn,
  get_device,
  parse_dataset_input,
  train,
  wandb_init_2,
)


def _broadcast_udq_control(p_flat: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
  """Broadcast a UDQ control to match flattened particles.

  p_flat: (B, 8). u: (8,), (1, 8), or (B, 8) — anything else raises.
  """
  if u.dim() == 1:
    return u.unsqueeze(0).expand(p_flat.shape[0], -1)
  if u.shape[0] == p_flat.shape[0]:
    return u
  if u.shape[0] == 1:
    return u.expand(p_flat.shape[0], -1)
  raise ValueError(f'Control shape {tuple(u.shape)} is incompatible with particles {tuple(p_flat.shape)}')


def _udq_log(udq: torch.Tensor) -> torch.Tensor:
  """UDQ (..,8) -> twist (..,6) = (ω, t) in decoupled chart.

  ω = 2·atan2(‖q_imag‖, q_w)·u over the q_w ≥ 0 hemisphere.
  t = imag(2 · q_dual · q_real*).
  """
  qr, qd = udq[..., :4], udq[..., 4:]
  sign = torch.where(qr[..., 0:1] < 0, -torch.ones_like(qr[..., 0:1]), torch.ones_like(qr[..., 0:1]))
  qr = qr * sign
  qd = qd * sign

  q_imag = qr[..., 1:]
  imag_norm = torch.norm(q_imag, dim=-1, keepdim=True)
  half_theta = torch.atan2(imag_norm, qr[..., 0:1].clamp(min=0))
  u = q_imag / imag_norm.clamp(min=1e-8)
  omega = 2 * half_theta * u
  small = imag_norm < 1e-8
  omega = torch.where(small.expand_as(omega), torch.zeros_like(omega), omega)

  t = (2 * q_mul(qd, q_conj(qr)))[..., 1:]

  return torch.cat([omega, t], dim=-1)


def _udq_exp(xi: torch.Tensor) -> torch.Tensor:
  """Twist (..,6) = (ω, t) -> UDQ (..,8) in decoupled chart."""
  omega = xi[..., :3]
  t = xi[..., 3:]
  theta = torch.norm(omega, dim=-1, keepdim=True)
  half_theta = theta / 2
  small = theta < 1e-8

  u = omega / theta.clamp(min=1e-8)
  q_imag = torch.sin(half_theta) * u
  q_imag = torch.where(small.expand_as(q_imag), torch.zeros_like(q_imag), q_imag)
  q_w = torch.cos(half_theta)
  q = torch.cat([q_w, q_imag], dim=-1)
  return udq_from_qt(q, t)


class DynamicsModel(nn.Module, ABC):

  @abstractmethod
  def forward(self, p: torch.Tensor, u: torch.Tensor) -> torch.Tensor: ...


class OdometryDynamicsModelSE3(DynamicsModel):
  """UDQ-native SE(3) odometry baseline. Stochasticity baked into forward.

  Adds anisotropic z / roll / pitch noise with reduced weight (z and roll/pitch
  get 1/10 of the in-plane scale by default). Log-Gaussian noise is sampled in
  se(3) and composed via udq_mul.
  """

  def __init__(self, std_t: float, std_r: float):
    """
    :param std_t: scalar translation std; z gets std_t * 0.1, x/y get std_t.
    :param std_r: scalar rotation std; roll/pitch get std_r * 0.1, yaw gets std_r.
    """
    super().__init__()
    self.std_t = std_t
    self.std_r = std_r
    diag_t = torch.tensor([std_t ** 2, std_t ** 2, (0.1 * std_t) ** 2])
    diag_r = torch.tensor([(0.1 * std_r) ** 2, (0.1 * std_r) ** 2, std_r ** 2])
    self.register_buffer('_cov_t_mat', torch.diag(diag_t))
    self.register_buffer('_cov_r_mat', torch.diag(diag_r))

  def forward(self, p: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    :param p: particles, shaped (..., N, 8) UDQ
    :param u: local SE(3) control, shaped (..., 8) or (..., 1, 8) UDQ
    :return: updated particles, shaped (..., N, 8) UDQ
    """
    if p.shape[-1] != 8:
      raise ValueError(f'Expected particles shape (...,8), got {tuple(p.shape)}')
    if u.shape[-1] != 8:
      raise ValueError(f'Expected control shape (...,8), got {tuple(u.shape)}')

    du = udq_log_gaussian(self._cov_t_mat, self._cov_r_mat, size=p.shape[:-1]).to(
      device=p.device, dtype=p.dtype
    )
    return udq_mul(udq_mul(p, u), du)


class _AffineCoupling(nn.Module):

  def __init__(self, dim: int, cond_dim: int, hidden_dim: int, mask: torch.Tensor, scale_clip: float):
    super().__init__()
    self.dim = dim
    self.scale_clip = scale_clip
    self.register_buffer('mask', mask)

    self.net = nn.Sequential(
      nn.Linear(dim + cond_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, dim * 2)
    )

  def forward(self, x: torch.Tensor, cond: torch.Tensor, inverse: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    mask = self._buffers['mask']
    assert isinstance(mask, torch.Tensor)

    masked = x * mask
    h = torch.cat([masked, cond], dim=-1)
    s, t = self.net(h).chunk(2, dim=-1)

    # Bound affine scaling to avoid exploding corrections.
    s = torch.tanh(s) * self.scale_clip
    inv_mask = 1.0 - mask

    if inverse:
      x_inv = masked + inv_mask * ((x - t) * torch.exp(-s))
      log_det_j = (inv_mask * -s).sum(dim=-1)
      return x_inv, log_det_j

    return masked + inv_mask * (x * torch.exp(s) + t)

class _CNFDynamics(Protocol):
  def compute_nll(self, p: torch.Tensor, u_odom: torch.Tensor, u_ref: torch.Tensor) -> torch.Tensor: ...

class ResidualRealNVPDynamicsSE3(DynamicsModel):
  """A RealNVP Model reshapes an input distribution with invertible layers and conditioned on the 
  input (the particle+current odometry). As it is invertible then it's very fast to compute the 
  exact NLL given a reference."""
  def __init__(
      self,
      layers: int,
      hidden_dim: int,
      sigma: float,
      alpha: float,
      scale_clip: float,
      norm_eps: float,
  ):
    super().__init__()

    self.alpha = alpha
    self.norm_eps = norm_eps

    self.register_buffer('base_std', torch.tensor(sigma))

    # 6D xi_u + ‖t_u‖ + ‖ω_u‖ + 6D xi_p = 14
    cond_dim = 14

    mask_a = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    mask_b = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    self.couplings = nn.ModuleList([
      _AffineCoupling(6, cond_dim, hidden_dim, mask_a if i % 2 == 0 else mask_b, scale_clip)
      for i in range(layers)
    ])

  def _build_conditioning(self, u_log: torch.Tensor, p_log: torch.Tensor) -> torch.Tensor:
    omega_norm = torch.norm(u_log[:, :3], dim=-1, keepdim=True)
    trans_norm = torch.norm(u_log[:, 3:], dim=-1, keepdim=True)
    return torch.cat([u_log, trans_norm, omega_norm, p_log], dim=-1)

  def forward(self, p: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    if p.shape[-1] != 8:
      raise ValueError(f'Expected particles shape (...,8), got {tuple(p.shape)}')
    if u.shape[-1] != 8:
      raise ValueError(f'Expected control shape (...,8), got {tuple(u.shape)}')

    p_flat = p.reshape(-1, 8)
    u_matched = _broadcast_udq_control(
      p_flat, u.reshape(-1, 8) if u.dim() > 1 else u)

    p_log = _udq_log(p_flat)
    u_log = _udq_log(u_matched)
    cond = self._build_conditioning(u_log, p_log)

    base_std = self._buffers['base_std']
    assert isinstance(base_std, torch.Tensor)
    eps = torch.randn_like(u_log) * base_std.to(device=u_log.device, dtype=u_log.dtype)
    delta_log = eps
    for coupling in self.couplings:
      delta_log = coupling(delta_log, cond)

    u_res_log = u_log + self.alpha * delta_log
    u_res = _udq_exp(u_res_log)
    p_next = udq_mul(p_flat, u_res)
    return p_next.reshape_as(p)

  def compute_nll(self, p: torch.Tensor, u_odom: torch.Tensor, u_ref: torch.Tensor) -> torch.Tensor:
    if p.shape[-1] != 8:
      raise ValueError(f'Expected particles shape (...,8), got {tuple(p.shape)}')

    p_flat = p.reshape(-1, 8)
    u_odom_flat = _broadcast_udq_control(
      p_flat, u_odom.reshape(-1, 8) if u_odom.dim() > 1 else u_odom)
    u_ref_flat = _broadcast_udq_control(
      p_flat, u_ref.reshape(-1, 8) if u_ref.dim() > 1 else u_ref)

    p_log = _udq_log(p_flat)
    u_odom_log = _udq_log(u_odom_flat)
    u_ref_log = _udq_log(u_ref_flat)

    cond = self._build_conditioning(u_odom_log, p_log)

    target_delta = (u_ref_log - u_odom_log) / self.alpha

    z = target_delta
    sum_log_det_j = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    for coupling in reversed(self.couplings):
      z, log_det_j = coupling(z, cond, inverse=True)
      sum_log_det_j = sum_log_det_j + log_det_j

    base_std = self._buffers['base_std']
    assert isinstance(base_std, torch.Tensor)
    base_std = base_std.to(device=z.device, dtype=z.dtype).clamp_min(self.norm_eps)

    nll = 0.5 * ((z / base_std) ** 2 + 2 * torch.log(base_std) + math.log(2 * math.pi))
    nll_z = nll.sum(dim=-1)

    exact_nll = nll_z - sum_log_det_j
    return exact_nll.mean()

  @property
  def sigma(self) -> float:
    base_std = self._buffers['base_std']
    assert isinstance(base_std, torch.Tensor)
    return float(base_std)


class AnchoredEnsembleRealNVPDynamicsSE3(DynamicsModel):

  def __init__(
      self,
      num_members: int,
      bootstrap: bool,
      anchor_lambda: float,
      layers: int,
      hidden_dim: int,
      sigma: float,
      alpha: float,
      scale_clip: float,
      norm_eps: float,
  ):
    super().__init__()
    self.num_members = num_members
    self.bootstrap = bootstrap
    self.anchor_lambda = anchor_lambda

    self.members = nn.ModuleList([
      ResidualRealNVPDynamicsSE3(
        layers=layers,
        hidden_dim=hidden_dim,
        sigma=sigma,
        alpha=alpha,
        scale_clip=scale_clip,
        norm_eps=norm_eps,
      )
      for _ in range(num_members)
    ])

    for m, member in enumerate(self.members):
      for i, param in enumerate(member.parameters()):
        self.register_buffer(f'anchor_m{m}_p{i}', param.detach().clone())

  def _anchor_penalty(self, m: int) -> torch.Tensor:
    member = self.members[m]
    first = next(member.parameters())
    penalty = torch.zeros((), device=first.device, dtype=first.dtype)
    for i, param in enumerate(member.parameters()):
      anchor = self.get_buffer(f'anchor_m{m}_p{i}')
      penalty = penalty + (param - anchor).pow(2).sum()
    return penalty

  def forward(self, p: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    if p.shape[-1] != 8:
      raise ValueError(f'Expected particles shape (...,8), got {tuple(p.shape)}')
    if u.shape[-1] != 8:
      raise ValueError(f'Expected control shape (...,8), got {tuple(u.shape)}')

    p_flat = p.reshape(-1, 8)
    u_flat = u.reshape(-1, 8) if u.dim() > 1 else u
    u_matched = _broadcast_udq_control(p_flat, u_flat)

    out = torch.empty_like(p_flat)
    for m, member in enumerate(self.members):
      idx = torch.arange(m, p_flat.shape[0], self.num_members, device=p_flat.device)
      if idx.numel() == 0:
        continue
      out[idx] = member(p_flat[idx], u_matched[idx])
    return out.reshape_as(p)

  def compute_nll(self, p: torch.Tensor, u_odom: torch.Tensor, u_ref: torch.Tensor) -> torch.Tensor:
    if p.shape[-1] != 8:
      raise ValueError(f'Expected particles shape (...,8), got {tuple(p.shape)}')

    p_flat = p.reshape(-1, 8)
    u_odom_flat = _broadcast_udq_control(
      p_flat, u_odom.reshape(-1, 8) if u_odom.dim() > 1 else u_odom)
    u_ref_flat = _broadcast_udq_control(
      p_flat, u_ref.reshape(-1, 8) if u_ref.dim() > 1 else u_ref)

    B = p_flat.shape[0]
    losses = []
    for m, member in enumerate(self.members):
      if self.bootstrap and self.training:
        idx = torch.randint(0, B, (B,), device=p_flat.device)
      else:
        idx = torch.arange(B, device=p_flat.device)

      nll = cast(_CNFDynamics, member).compute_nll(p_flat[idx], u_odom_flat[idx], u_ref_flat[idx])
      penalty = self.anchor_lambda * self._anchor_penalty(m)
      losses.append(nll + penalty)

    return torch.stack(losses).mean()


class _MLPMember(nn.Module):
  """Plain MLP head: cond -> mean delta. Used by anchored ensemble dynamics."""

  def __init__(self, cond_dim: int, out_dim: int, hidden_dim: int, layers: int):
    super().__init__()
    blocks: list[nn.Module] = []
    last = cond_dim
    for _ in range(layers):
      blocks += [nn.Linear(last, hidden_dim), nn.ReLU()]
      last = hidden_dim
    blocks += [nn.Linear(last, out_dim)]
    self.net = nn.Sequential(*blocks)

  def forward(self, cond: torch.Tensor) -> torch.Tensor:
    return self.net(cond)


class AnchoredEnsembleMLPDynamicsSE3(DynamicsModel):
  """Plain anchored ensemble of MLPs over the 6D (ω, t) twist chart."""

  def __init__(
      self,
      num_members: int,
      bootstrap: bool,
      anchor_lambda: float,
      layers: int,
      hidden_dim: int,
      sigma_omega: float,
      sigma_v: float,
      alpha: float,
      norm_eps: float,
  ):
    super().__init__()
    self.num_members = num_members
    self.bootstrap = bootstrap
    self.anchor_lambda = anchor_lambda
    self.alpha = alpha
    self.norm_eps = norm_eps

    self.register_buffer('base_std', torch.tensor([
      sigma_omega/10, sigma_omega/10, sigma_omega, sigma_v, sigma_v, sigma_v/10,
    ]))

    # Same conditioning as ResidualRealNVPDynamicsSE3: u_log(6) + ‖t‖(1) + ‖ω‖(1) + p_log(6).
    cond_dim = 14

    self.members = nn.ModuleList([
      _MLPMember(cond_dim, 6, hidden_dim, layers) for _ in range(num_members)
    ])

    for m, member in enumerate(self.members):
      for i, param in enumerate(member.parameters()):
        self.register_buffer(f'anchor_m{m}_p{i}', param.detach().clone())

  def _anchor_penalty(self, m: int) -> torch.Tensor:
    member = self.members[m]
    first = next(member.parameters())
    penalty = torch.zeros((), device=first.device, dtype=first.dtype)
    for i, param in enumerate(member.parameters()):
      anchor = self.get_buffer(f'anchor_m{m}_p{i}')
      penalty = penalty + (param - anchor).pow(2).sum()
    return penalty

  @staticmethod
  def _build_conditioning(u_log: torch.Tensor, p_log: torch.Tensor) -> torch.Tensor:
    omega_norm = torch.norm(u_log[:, :3], dim=-1, keepdim=True)
    trans_norm = torch.norm(u_log[:, 3:], dim=-1, keepdim=True)
    return torch.cat([u_log, trans_norm, omega_norm, p_log], dim=-1)

  def forward(self, p: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    if p.shape[-1] != 8:
      raise ValueError(f'Expected particles shape (...,8), got {tuple(p.shape)}')
    if u.shape[-1] != 8:
      raise ValueError(f'Expected control shape (...,8), got {tuple(u.shape)}')

    p_flat = p.reshape(-1, 8)
    u_matched = _broadcast_udq_control(
      p_flat, u.reshape(-1, 8) if u.dim() > 1 else u)

    p_log = _udq_log(p_flat)
    u_log = _udq_log(u_matched)
    cond = self._build_conditioning(u_log, p_log)

    base_std = self._buffers['base_std']
    assert isinstance(base_std, torch.Tensor)
    base_std = base_std.to(device=u_log.device, dtype=u_log.dtype)

    out = torch.empty_like(p_flat)
    for m, member in enumerate(self.members):
      idx = torch.arange(m, p_flat.shape[0], self.num_members, device=p_flat.device)
      if idx.numel() == 0:
        continue
      mu = member(cond[idx])
      eps = torch.randn_like(mu) * base_std
      delta = mu + eps
      u_res_log = u_log[idx] + self.alpha * delta
      out[idx] = udq_mul(p_flat[idx], _udq_exp(u_res_log))

    return out.reshape_as(p)

  def compute_nll(self, p: torch.Tensor, u_odom: torch.Tensor, u_ref: torch.Tensor) -> torch.Tensor:
    if p.shape[-1] != 8:
      raise ValueError(f'Expected particles shape (...,8), got {tuple(p.shape)}')

    p_flat = p.reshape(-1, 8)
    u_odom_flat = _broadcast_udq_control(
      p_flat, u_odom.reshape(-1, 8) if u_odom.dim() > 1 else u_odom)
    u_ref_flat = _broadcast_udq_control(
      p_flat, u_ref.reshape(-1, 8) if u_ref.dim() > 1 else u_ref)

    p_log = _udq_log(p_flat)
    u_odom_log = _udq_log(u_odom_flat)
    u_ref_log = _udq_log(u_ref_flat)
    cond = self._build_conditioning(u_odom_log, p_log)

    target_delta = (u_ref_log - u_odom_log) / self.alpha

    base_std = self._buffers['base_std']
    assert isinstance(base_std, torch.Tensor)
    base_std = base_std.to(device=cond.device, dtype=cond.dtype).clamp_min(self.norm_eps)

    B = p_flat.shape[0]
    losses = []
    for m, member in enumerate(self.members):
      if self.bootstrap and self.training:
        idx = torch.randint(0, B, (B,), device=p_flat.device)
      else:
        idx = torch.arange(B, device=p_flat.device)

      mu = member(cond[idx])
      residual = target_delta[idx] - mu
      nll = 0.5 * ((residual / base_std) ** 2 + 2 * torch.log(base_std) + math.log(2 * math.pi))
      nll_mean = nll.sum(dim=-1).mean()
      penalty = self.anchor_lambda * self._anchor_penalty(m)
      losses.append(nll_mean + penalty)

    return torch.stack(losses).mean()


class DynamicsDataset(LocalisationDataset):

  def __len__(self):
    return super().__len__() - 1

  def __getitem__(self, idx):
    p0 = self.poses[idx]
    p1 = self.poses[idx + 1]
    o0 = self.odometry[idx]
    o1 = self.odometry[idx + 1]
    u_odom = udq_mul(udq_conj(o0), o1)
    u_ref = udq_mul(udq_conj(p0), p1)
    return p0.unsqueeze(0), u_odom.unsqueeze(0), u_ref.unsqueeze(0)

def loss_fn_cnf(dynamics: nn.Module, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], args: Any) -> torch.Tensor:
  """loss for training condtional normalizing flow (RealNVP)"""
  p0, u_odom, u_ref = batch
  return cast(_CNFDynamics, dynamics).compute_nll(p0, u_odom, u_ref)


@dataclass
class DynamicsConfig(ModuleConfig):
  """Base for motion-model configs.

  Overrides weight loading to be strict: a config rebuilds the exact
  architecture, so any state-dict key mismatch signals config/checkpoint drift
  rather than an intended partial load.
  """

  @staticmethod
  def apply_state_dict(model: nn.Module, path: Path):
    data = torch.load(path, map_location='cpu')
    model.load_state_dict(data['state_dict'], strict=True)


@dataclass
@ConfigRegistry.register
class OdometrySE3Config(DynamicsConfig):
  std_t: float = 0.2
  """translation std (x/y); z gets std_t * 0.1"""

  std_r: float = 0.15
  """rotation std (yaw); roll/pitch get std_r * 0.1"""

  def build_model(self) -> 'OdometryDynamicsModelSE3':
    return OdometryDynamicsModelSE3(std_t=self.std_t, std_r=self.std_r)


@dataclass
@ConfigRegistry.register
class RealNVPConfig(DynamicsConfig):
  layers: int = 4
  """coupling layers"""

  hidden_dim: int = 64
  """coupling-network hidden size"""

  sigma: float = 0.1
  """isotropic base-noise scale"""

  alpha: float = 0.3
  """residual-correction weight"""

  scale_clip: float = 1.5
  """bound on affine log-scale"""

  norm_eps: float = 1e-6
  """base-std floor in the NLL"""

  def build_model(self) -> 'ResidualRealNVPDynamicsSE3':
    return ResidualRealNVPDynamicsSE3(
      layers=self.layers, hidden_dim=self.hidden_dim, sigma=self.sigma,
      alpha=self.alpha, scale_clip=self.scale_clip, norm_eps=self.norm_eps,
    )


@dataclass
@ConfigRegistry.register
class EnsembleConfig(DynamicsConfig):
  num_members: int = 20
  """ensemble members"""

  bootstrap: bool = True
  """bootstrap-resample each member's batch during training"""

  anchor_lambda: float = 1e-4
  """anchor (weight-prior) penalty weight"""

  layers: int = 4
  hidden_dim: int = 64
  sigma: float = 0.1
  alpha: float = 0.3
  scale_clip: float = 1.5
  norm_eps: float = 1e-6

  def build_model(self) -> 'AnchoredEnsembleRealNVPDynamicsSE3':
    return AnchoredEnsembleRealNVPDynamicsSE3(
      num_members=self.num_members, bootstrap=self.bootstrap,
      anchor_lambda=self.anchor_lambda, layers=self.layers,
      hidden_dim=self.hidden_dim, sigma=self.sigma, alpha=self.alpha,
      scale_clip=self.scale_clip, norm_eps=self.norm_eps,
    )


@dataclass
@ConfigRegistry.register
class MLPConfig(DynamicsConfig):
  num_members: int = 5
  bootstrap: bool = True
  anchor_lambda: float = 1e-4
  layers: int = 2
  hidden_dim: int = 64

  sigma_omega: float = 0.02
  """yaw-rate noise std; roll/pitch get sigma_omega / 10"""

  sigma_v: float = 0.06
  """forward/lateral velocity noise std; z gets sigma_v / 10"""

  alpha: float = 0.3
  norm_eps: float = 1e-6

  def build_model(self) -> 'AnchoredEnsembleMLPDynamicsSE3':
    return AnchoredEnsembleMLPDynamicsSE3(
      num_members=self.num_members, bootstrap=self.bootstrap,
      anchor_lambda=self.anchor_lambda, layers=self.layers,
      hidden_dim=self.hidden_dim, sigma_omega=self.sigma_omega,
      sigma_v=self.sigma_v, alpha=self.alpha, norm_eps=self.norm_eps,
    )


# CLI mode string -> config (odometry-se3 is a parameter-free baseline).
DYNAMICS_CONFIGS: dict[str, type[DynamicsConfig]] = {
  'odometry-se3': OdometrySE3Config,
  'realnvp-se3': RealNVPConfig,
  'ensemble-se3': EnsembleConfig,
  'mlp-se3': MLPConfig,
}

MOTION_MODES = tuple(DYNAMICS_CONFIGS)

_MODE_BY_MODEL = {
  'OdometryDynamicsModelSE3': 'odometry-se3',
  'ResidualRealNVPDynamicsSE3': 'realnvp-se3',
  'AnchoredEnsembleRealNVPDynamicsSE3': 'ensemble-se3',
  'AnchoredEnsembleMLPDynamicsSE3': 'mlp-se3',
}


def mode_from_dynamics(model: 'DynamicsModel') -> str:
  """Recover the CLI mode string from a model instance."""
  name = type(model).__name__
  if name not in _MODE_BY_MODEL:
    raise ValueError(f'Unknown dynamics class: {name}')
  return _MODE_BY_MODEL[name]


def build_dynamics_config(
    mode: str,
    std_t: float = 0.10,
    std_r: float = 0.05,
    layers: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    num_members: Optional[int] = None,
) -> DynamicsConfig:
  """Build a fresh motion-model config from CLI-style args. `std_t`/`std_r` map
  to odometry/MLP noise scales (sigma_v/sigma_omega) and to the RealNVP base
  `sigma` (via `std_t`). `layers`/`hidden_dim`/`num_members` override per-config
  defaults when not None. Used by the dpf CLI's fresh --motion-arch path and by
  eval; the dynamics CLI builds configs directly.
  """
  if mode == 'odometry-se3':
    return OdometrySE3Config(std_t=std_t, std_r=std_r)
  if mode == 'realnvp-se3':
    cfg = RealNVPConfig(sigma=std_t)
    if layers is not None: cfg.layers = layers
    if hidden_dim is not None: cfg.hidden_dim = hidden_dim
    return cfg
  if mode == 'ensemble-se3':
    cfg = EnsembleConfig()
    if layers is not None: cfg.layers = layers
    if hidden_dim is not None: cfg.hidden_dim = hidden_dim
    if num_members is not None: cfg.num_members = num_members
    return cfg
  if mode == 'mlp-se3':
    cfg = MLPConfig(sigma_omega=std_r, sigma_v=std_t)
    if layers is not None: cfg.layers = layers
    if hidden_dim is not None: cfg.hidden_dim = hidden_dim
    if num_members is not None: cfg.num_members = num_members
    return cfg
  raise ValueError(f'Unknown dynamics mode: {mode!r}; choose from {MOTION_MODES}')


@dataclass
class DynamicsTrainingParams:
  train_ds: str = 'data/street,init'
  """Training datasets. Format as 'data/DATASET_NAME,BAG1,BAG2,...'"""

  val_ds: str = 'data/street,d1'
  """Validation datasets, same format as train_ds."""

  save_tag: str = ''
  """Optional suffix on the models/dynamics-<arch>[-<tag>] output directory."""

  epochs: int = 100
  lr: float = 1e-3
  batch_size: int = 64
  lidar_topic: str = '/livox/lidar'
  odometry_topic: str = '/odometry/filtered'


@dataclass
class TrainDynamics:
  """Train a motion (dynamics) model."""

  model: RealNVPConfig | EnsembleConfig | MLPConfig
  train: DynamicsTrainingParams = field(default_factory=DynamicsTrainingParams)
  wandb: WBConfig = field(default_factory=WBConfig)


_MODE_BY_CONFIG = {v: k for k, v in DYNAMICS_CONFIGS.items()}


def train_dynamics(cmd: TrainDynamics):
  device = get_device()
  p = cmd.train

  ds_kwargs = {'lidar_topic': p.lidar_topic, 'odometry_topic': p.odometry_topic}
  dl_kwargs = {'batch_size': p.batch_size, 'shuffle': True, 'collate_fn': collate_fn}

  train_ds = ConcatDataset([DynamicsDataset(d, **ds_kwargs) for d in parse_dataset_input(p.train_ds)])
  val_ds = ConcatDataset([DynamicsDataset(d, **ds_kwargs) for d in parse_dataset_input(p.val_ds)])
  tdl = DataLoader(train_ds, **dl_kwargs)
  vdl = DataLoader(val_ds, **dl_kwargs)

  wandb_init_2(cmd.wandb, asdict(cmd))

  model = cmd.model.build_model()
  model.to(device)

  mode = _MODE_BY_CONFIG[type(cmd.model)]
  save_tag = f'dynamics-{mode}' + (f'-{p.save_tag}' if p.save_tag else '')

  optimizer = torch.optim.Adam(model.parameters(), lr=p.lr, weight_decay=1e-5)
  result = train(model, optimizer, loss_fn_cnf, p.epochs, tdl, vdl, device, save_tag=save_tag, patience=10, config=cmd.model)
  if wandb.run is not None:
    wandb.finish()
  return result


if __name__ == '__main__':
  train_dynamics(tyro.cli(TrainDynamics))
