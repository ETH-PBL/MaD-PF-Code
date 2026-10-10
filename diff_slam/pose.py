from typing import Iterable

import torch
from torch.distributions import MultivariateNormal


def q_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  w1, x1, y1, z1 = q1.unbind(dim=-1)
  w2, x2, y2, z2 = q2.unbind(dim=-1)
  return torch.stack([
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
  ], dim=-1)


def q_conj(q: torch.Tensor) -> torch.Tensor:
  return q * torch.tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype)


def udq_mul(udq1: torch.Tensor, udq2: torch.Tensor) -> torch.Tensor:
  """
  unit dual quaternion multiplication, equivalent to matrix multiplication T1 @ T2 for SE(3) transforms

  :param udq1: first transform, shaped (..., 8)
  :param udq2: second transform, shaped (..., 8)
  :return: composed udq, shaped (..., 8)
  """
  qr1, qd1 = udq1[..., :4], udq1[..., 4:]
  qr2, qd2 = udq2[..., :4], udq2[..., 4:]

  return torch.cat([
    q_mul(qr1, qr2),
    q_mul(qr1, qd2) + q_mul(qd1, qr2)
  ], dim=-1)


def udq_conj(udq: torch.Tensor) -> torch.Tensor:
  return udq * torch.tensor([1, -1, -1, -1, 1, -1, -1, -1], device=udq.device, dtype=udq.dtype)


def udq_transform_points(udq: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
  """
  apply a unit dual quaternion transform to a point set

  :param udq: udq tensor representing the SE(3) transform, shaped (..., 8)
  :param points: point set tensor, shaped (..., N, M) with M >= 3, the first 3 components are treated as xyz
  :return: transformed point set, shaped (..., N, M)
  """
  qr = udq[..., :4].unsqueeze(-2)
  qd = udq[..., 4:].unsqueeze(-2)

  # convert points to pure quaternion
  pq = torch.cat([
    torch.zeros_like(points[..., :1]),
    points[..., :3]
  ], dim=-1)

  pq = q_mul(q_mul(qr, pq), q_conj(qr))
  t = 2.0 * q_mul(qd, q_conj(qr))

  return torch.cat([
    pq[..., 1:] + t[..., 1:],
    points[..., 3:].expand(*torch.broadcast_shapes(points.shape[:-2], udq.shape[:-1]), -1, -1)
  ], dim=-1)


@torch.compile(mode='max-autotune-no-cudagraphs')
def udq_transform_points_fast(udq: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
  # udq: [wr, u, wd, d]
  wr, r, wd, d = udq[..., :1], udq[..., 1:4], udq[..., 4:5], udq[..., 5:]

  # rodrigues rotation formula
  pos = points[..., :3]
  r_ = r.unsqueeze(-2).expand_as(pos)
  rp = torch.cross(r_, pos, dim=-1)
  pos = pos + 2.0 * (wr.unsqueeze(-2) * rp + torch.cross(r_, rp, dim=-1))

  # apply translation
  pos = pos + (2.0 * (wr * d - wd * r + torch.cross(r, d, dim=-1))).unsqueeze(-2)

  if points.shape[-1] > 3:
    return torch.cat([pos, points[..., 3:]], dim=-1)

  return pos


def udq_distance(udq1: torch.Tensor, udq2: torch.Tensor, a: float = 1) -> torch.Tensor:
  """
  differentiable, singularity-free distance function between unit dual quaternions

  :param udq1: udq tensor, shaped (..., 8)
  :param udq2: udq tensor, shaped (..., 8), note that this function is symmetric
  :param a: translation / rotation weight coefficient
  :return: scalar distance values shaped (...)
  """
  qr1, qd1 = udq1[..., :4], udq1[..., 4:]
  qr2, qd2 = udq2[..., :4], udq2[..., 4:]

  # chordal rotation distance
  dist_r = 1.0 - torch.sum(qr1 * qr2, dim=-1).abs()

  # L2 squared
  t1 = 2.0 * q_mul(qd1, q_conj(qr1))[..., 1:]
  t2 = 2.0 * q_mul(qd2, q_conj(qr2))[..., 1:]
  dist_t = torch.sum((t1 - t2) ** 2, dim=-1)

  return dist_r + a * dist_t


def udq_mean(poses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
  """
  weighted mean computation for unit dual quaternions using linear blending
  operation is fully differentiable and handles the antipodal / double-cover singularity

  :param poses: udq tensor, shaped (..., N, 8)
  :param weights: scalar weight tensor, shaped (..., N, 1), assumed to be normalized along dim=-2
  :return: mean udq tensor, shaped (..., 8)
  """
  qr0 = poses[..., 0:1, :4]  # reference orientation
  dot = torch.sum(poses[..., :4] * qr0, dim=-1, keepdim=True)

  sign = torch.where(dot < 0, -1.0, 1.0).detach() # zero proof sign

  mean_pose = torch.sum(weights * poses * sign, dim=-2)

  # enforce norm constraint: |qr| = 1
  norm_qr = torch.norm(mean_pose[..., :4], dim=-1, keepdim=True)
  mean_pose = mean_pose / torch.clip(norm_qr, min=1e-8)

  # enforce orthogonality: dot(qr, qd) = 0
  dot = torch.sum(mean_pose[..., :4] * mean_pose[..., 4:], dim=-1, keepdim=True)
  qd_orthogonal = mean_pose[..., 4:] - dot * mean_pose[..., :4]

  return torch.cat([mean_pose[..., :4], qd_orthogonal], dim=-1)


def udq_from_qt(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
  """
  derive a unit dual quaternion from a unit quaternion and translation vector

  :param q: rotation quaternion, shaped (..., 4), must be a unit quaternion
  :param t: translation vector, shaped (..., 3)
  :return: udq tensor, shaped (..., 8)
  """
  qt = torch.cat([torch.zeros_like(t[..., :1]), t], dim=-1)
  return torch.cat([q, 0.5 * q_mul(qt, q)], dim=-1)


def udq_log_gaussian(cov_t: torch.Tensor, cov_r: torch.Tensor, size: torch.Size | Iterable[int]) -> torch.Tensor:
  """
  draw samples from a log-Gaussian distribution on SE(3)

  :param cov_t: translation covariance matrix, shaped (3, 3)
  :param cov_r: rotation covariance matrix, shaped (3, 3)
  :param size: shape of samples to take
  :return: tensor containing udq samples, shaped (size, 8)
  """
  dt = MultivariateNormal(torch.zeros(3, device=cov_t.device, dtype=cov_t.dtype), cov_t).sample(size)
  dw = MultivariateNormal(torch.zeros(3, device=cov_r.device, dtype=cov_r.dtype), cov_r).sample(size)

  # exp map for SO(3)
  theta = torch.norm(dw, dim=-1, keepdim=True)
  u = dw / torch.clamp(theta, min=1e-8)

  q = torch.cat([
    torch.cos(theta / 2.0),
    torch.sin(theta / 2.0) * u
  ], dim=-1)

  return udq_from_qt(q, dt)


def udq_to_matrix(udq: torch.Tensor) -> torch.Tensor:
  qr, qd = udq[..., :4], udq[..., 4:]

  T = torch.eye(4, device=udq.device, dtype=udq.dtype).expand(*udq.shape[:-1], 4, 4).clone()

  w, x, y, z = qr.unbind(dim=-1)
  T[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
  T[..., 0, 1] = 2.0 * (x * y - z * w)
  T[..., 0, 2] = 2.0 * (x * z + y * w)
  T[..., 1, 0] = 2.0 * (x * y + z * w)
  T[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
  T[..., 1, 2] = 2.0 * (y * z - x * w)
  T[..., 2, 0] = 2.0 * (x * z - y * w)
  T[..., 2, 1] = 2.0 * (y * z + x * w)
  T[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)

  t = 2.0 * q_mul(qd, q_conj(qr))
  T[..., :3, 3] = t[..., 1:]

  return T
