from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch

from .module import ConfigRegistry
from .pose import udq_mean, udq_distance


ResamplingStrategy = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class ResampleConfig(ABC):

  @abstractmethod
  def get_strategy(self) -> ResamplingStrategy: ...


@dataclass
@ConfigRegistry.register
class ContractResampleConfig(ResampleConfig):
  f: float = 0.8
  """contraction factor between 0 (none) and 1 (full)"""

  def get_strategy(self) -> ResamplingStrategy:
    return partial(contract_resample, f=self.f)


def contract_resample(particles: torch.Tensor, weights: torch.Tensor, f: float = 0.8) -> torch.Tensor:
  """
  Mean-based contraction resampling

  :param particles: tensor shaped (..., N, D) representing N particles in D dimensions
  :param weights: (..., N, 1) weight tensor
  :param f: contraction factor between 0 (none) and 1 (full)
  :return: resampled (..., N, D) particle tensor
  """
  mean = udq_mean(particles, weights)
  fw = torch.tensor([1 - f, f], device=particles.device)

  return udq_mean(torch.stack([
    particles,
    mean.expand_as(particles)
  ], dim=-2), fw.unsqueeze(-1))


@dataclass
@ConfigRegistry.register
class GSResampleConfig(ResampleConfig):
  tau: float = 0.5
  """softmax temperature"""

  hard: bool = False
  """use hard sampling with pass through gradients"""

  def get_strategy(self) -> ResamplingStrategy:
    return partial(gumbel_softmax_resample, tau=self.tau, hard=self.hard)


def gumbel_softmax_resample(particles: torch.Tensor, weights: torch.Tensor, tau: float = 0.5, hard: bool = False) -> torch.Tensor:
  """
  Differentiable resampling using continuous relaxation (Gumbel-Softmax) for UDQs

  :param particles: tensor shaped (N, 8) representing N particles in SE(3)
  :param weights: weight tensor shaped (N, ) or (N, 1) containing N scalars
  :param tau: softmax temperature
  :param hard: use hard sampling with pass through gradients
  :return: resampled (N, 8) particle tensor
  """
  N = particles.shape[0]

  # N sets of resampling weights using Gumbel-Softmax
  logits = torch.log(weights.squeeze(-1) + 1e-9).unsqueeze(0).expand(N, -1)
  weights = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

  return udq_mean(
    particles.unsqueeze(0).expand(N, -1, -1),  # broadcast (N, N, 8)
    weights.unsqueeze(-1)
  )


def sinkhorn_knopp(C: torch.Tensor, a: torch.Tensor, b: torch.Tensor, epsilon: float = 0.05, n_iters: int = 50) -> torch.Tensor:
  K = torch.exp(-C / epsilon)
  u = torch.ones_like(a)
  v = torch.ones_like(b)

  for _ in range(n_iters):
    v = b / (K.t() @ u + 1e-9)
    u = a / (K @ v + 1e-9)

  return u.unsqueeze(1) * K * v.unsqueeze(0)


@dataclass
@ConfigRegistry.register
class OTResampleConfig(ResampleConfig):
  epsilon: float = 0.05
  """entropy regularization"""

  a: float = 1.0
  """translation / rotation weight coefficient for the distance metric"""

  def get_strategy(self) -> ResamplingStrategy:
    return partial(ot_resample, epsilon=self.epsilon, a=self.a)


def ot_resample(particles: torch.Tensor, weights: torch.Tensor, epsilon: float = 0.05, a: float = 1.0) -> torch.Tensor:
  """
  Differentiable resampling using Optimal Transport for UDQs

  :param particles: tensor shaped (N, 8) representing N particles in SE(3)
  :param weights: weight tensor shaped (N, ) or (N, 1) containing N scalars
  :param epsilon: entropy regularization
  :param a: translation / rotation weight coefficient for the distance metric
  :return: resampled (N, 8) particle tensor
  """
  N = particles.shape[0]

  # pairwise UDQ distance matrix
  C = udq_distance(
    particles.unsqueeze(1),
    particles.unsqueeze(0),
    a=a
  )

  C = C / (C.max() + 1e-9)  # normalize

  target_weights = torch.ones(N, dtype=particles.dtype, device=particles.device) / N
  ot = sinkhorn_knopp(C, target_weights, weights.squeeze(-1), epsilon=epsilon, n_iters=40)

  return udq_mean(
    particles.unsqueeze(0).expand(N, -1, -1),  # broadcast (N, N, 8)
    (N * ot).unsqueeze(-1)  # transport plan rows sum to 1/N -> scale to normalize to 1
  )


# CLI resample-mode string -> config.
RESAMPLE_CONFIGS: dict[str, type[ResampleConfig]] = {
  'contract': ContractResampleConfig,
  'gumbel': GSResampleConfig,
  'ot': OTResampleConfig,
}
