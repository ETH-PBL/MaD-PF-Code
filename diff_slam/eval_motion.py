"""Motion-model evaluation: render per-step motion-model snapshots from DPF checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import cast

import matplotlib
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from torch import nn
from tqdm import tqdm

from .bag_parser import LocalisationDataset
from .module import ModuleConfig
from .dpf import SequenceDataset, DPFConfig
from .dynamics import (
  AnchoredEnsembleMLPDynamicsSE3,
  AnchoredEnsembleRealNVPDynamicsSE3,
  DynamicsConfig,
  OdometryDynamicsModelSE3,
  ResidualRealNVPDynamicsSE3,
)
from .pose import udq_conj, udq_mul, udq_to_matrix
from .util import get_device


BASELINE_CHOICES = ('odometry-se3',)

# Classes whose forward operates on UDQ particles (..., N, 8). Other SE(2) 4x4
# flow variants are not supported by this UDQ-native evaluator.
UDQ_MOTION_TYPES: tuple[type, ...] = (
  OdometryDynamicsModelSE3,
  ResidualRealNVPDynamicsSE3,
  AnchoredEnsembleRealNVPDynamicsSE3,
  AnchoredEnsembleMLPDynamicsSE3,
)

SE3_MOTION_TYPES: tuple[type, ...] = (
  OdometryDynamicsModelSE3,
  ResidualRealNVPDynamicsSE3,
  AnchoredEnsembleRealNVPDynamicsSE3,
  AnchoredEnsembleMLPDynamicsSE3,
)

matplotlib.use('Agg')


DEFAULT_EVAL_DATASETS = [
  Path('data/jackal-v2/d9'),
  Path('data/jackal-v2/d10'),
  Path('data/jackal-v2/s9'),
]


def _safe_name(path: Path) -> str:
  return str(path).replace(os.sep, '__').replace('/', '__')


def _make_run_dir(output_root: Path) -> Path:
  ts = datetime.now().strftime('%Y%m%d_%H%M%S')
  run_dir = output_root / ts
  run_dir.mkdir(parents=True, exist_ok=False)
  return run_dir


def _create_checkpoint_artifact_dir(run_dir: Path, checkpoint: Path) -> Path:
  out = run_dir / _safe_name(checkpoint)
  out.mkdir(parents=True, exist_ok=True)
  return out


def _create_model_symlink(artifact_dir: Path, checkpoint: Path) -> str:
  link_path = artifact_dir / 'model.pt'
  if link_path.exists() or link_path.is_symlink():
    link_path.unlink()
  os.symlink(checkpoint.resolve(), link_path)
  return str(link_path)


def _serialize_arg_value(value: object) -> object:
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, list):
    return [_serialize_arg_value(v) for v in value]
  return value


def _save_run_metadata(
  run_dir: Path,
  args: argparse.Namespace,
  device: torch.device,
  checkpoints: list[Path],
):
  payload = {
    'timestamp': run_dir.name,
    'device': str(device),
    'args': {k: _serialize_arg_value(v) for k, v in vars(args).items()},
    'checkpoints': [str(p) for p in checkpoints],
  }
  (run_dir / 'run_metadata.json').write_text(json.dumps(payload, indent=2))


def _epoch_sort_key(path: Path) -> tuple[int, int, str]:
  name = path.name
  if name == 'model.pt':
    return 0, -1, name
  if name.startswith('epoch-') and name.endswith('.pt'):
    try:
      return 1, int(name[len('epoch-'):-len('.pt')]), name
    except ValueError:
      pass
  return 2, 0, name


def discover_motion_checkpoints(models_dir: Path, max_checkpoints: int = 0) -> list[Path]:
  dpf_dir = models_dir
  if not dpf_dir.exists():
    return []
  ckpts = sorted(
    [p for p in dpf_dir.glob('*.pt') if p.name != 'loss.pt'],
    key=_epoch_sort_key,
  )
  if max_checkpoints > 0:
    ckpts = ckpts[:max_checkpoints]
  return ckpts


def select_indices(length: int, count: int) -> list[int]:
  if length <= 0:
    return []
  if count == 1:
    return [length // 2]
  if count <= 0 or length <= count:
    return list(range(length))
  step = (length - 1) / (count - 1)
  idx = {int(round(i * step)) for i in range(count)}
  return sorted(idx)


def motion_snapshots_udq(
    motion_model: nn.Module,
    particle_count: int,
    x_ref: torch.Tensor,
    odometry: torch.Tensor,
) -> torch.Tensor:
  """Per-step motion-model spread, UDQ-native, seeded fresh from x_ref each step.

  Args:
    x_ref: (T, 8) UDQ reference poses.
    odometry: (T, 8) UDQ odometry poses.

  Returns:
    (T-1, N, 8) UDQ particles after one motion_model step from x_ref[t].
  """
  if x_ref.shape[-1] != 8 or odometry.shape[-1] != 8:
    raise ValueError(f'Expected UDQ inputs (...,8); got x_ref={tuple(x_ref.shape)} odometry={tuple(odometry.shape)}')

  u = udq_mul(udq_conj(odometry[:-1]), odometry[1:])  # (T-1, 8)
  snapshots = []
  for t in range(u.shape[0]):
    p = x_ref[t].unsqueeze(0).expand(particle_count, -1).contiguous()  # (N, 8)
    p = motion_model(p, u[t])
    snapshots.append(p.detach().clone())
  return torch.stack(snapshots)  # (T-1, N, 8)


# Triangle marker for top-down x-y view.
_MOTION_TRIANGLE = torch.tensor([
  [0.30, 0.00],
  [-0.20, 0.15],
  [-0.20, -0.15],
]) * 0.15


def _percentile_bounds(
    vals_a: torch.Tensor,
    vals_b: torch.Tensor,
    gt_a: float,
    gt_b: float,
    bounds_pct: float,
    bounds_pad: float,
) -> tuple[float, float, float, float]:
  """Adaptive symmetric bounds covering particles + ground truth + origin."""
  q = max(0.0, min(bounds_pct / 100.0, 1.0))
  q_lo, q_hi = (1.0 - q) / 2.0, 1.0 - (1.0 - q) / 2.0
  a_lo, a_hi = float(torch.quantile(vals_a, q_lo).item()), float(torch.quantile(vals_a, q_hi).item())
  b_lo, b_hi = float(torch.quantile(vals_b, q_lo).item()), float(torch.quantile(vals_b, q_hi).item())

  a_lo, a_hi = min(0.0, a_lo, gt_a), max(0.0, a_hi, gt_a)
  b_lo, b_hi = min(0.0, b_lo, gt_b), max(0.0, b_hi, gt_b)

  pad_a = max(a_hi - a_lo, 0.1) * max(bounds_pad, 0.0)
  pad_b = max(b_hi - b_lo, 0.1) * max(bounds_pad, 0.0)
  return a_lo - pad_a, a_hi + pad_a, b_lo - pad_b, b_hi + pad_b


def _render_xy(
    ax,
    p_local: torch.Tensor,
    g_local: torch.Tensor,
    bounds_pct: float,
    bounds_pad: float,
    step_idx: int,
    n_steps: int,
) -> int:
  """Draw a top-down (x-y) motion frame. Returns particle count drawn."""
  ax.clear()
  n_total = p_local.shape[0]
  px = p_local[:, 0, 3].float()
  py = p_local[:, 1, 3].float()
  g = g_local.cpu()
  gx, gy = float(g[0, 3].item()), float(g[1, 3].item())

  x_lo, x_hi, y_lo, y_hi = _percentile_bounds(px, py, gx, gy, bounds_pct, bounds_pad)

  if n_total > 0:
    particle_polys = [
      Polygon(_MOTION_TRIANGLE @ x[:2, :2].T + x[:2, 3], closed=True)
      for x in p_local.cpu()
    ]
    pc = PatchCollection(particle_polys, alpha=0.1)
    pc.set_color('crimson')
    ax.add_collection(pc)

  gt_poly = Polygon(_MOTION_TRIANGLE @ g[:2, :2].T + g[:2, 3], closed=True)
  gt_pc = PatchCollection([gt_poly])
  gt_pc.set_color('black')
  ax.add_collection(gt_pc)

  origin_pc = PatchCollection([Polygon(_MOTION_TRIANGLE, closed=True)])
  origin_pc.set_color('royalblue')
  ax.add_collection(origin_pc)

  ax.set_xlim(x_lo, x_hi)
  ax.set_ylim(y_lo, y_hi)
  ax.set_aspect('equal')
  ax.grid(True)
  ax.set_xlabel('dx (m, local frame)')
  ax.set_ylabel('dy (m, local frame)')
  ax.set_title(f'x-y · step {step_idx + 1}/{n_steps} (n={n_total})')
  return n_total


def _render_xz(
    ax,
    p_local: torch.Tensor,
    g_local: torch.Tensor,
    bounds_pct: float,
    bounds_pad: float,
    step_idx: int,
    n_steps: int,
):
  """Draw side-view (x-z) motion frame using scatter markers (orientation omitted)."""
  ax.clear()
  px = p_local[:, 0, 3].float().cpu()
  pz = p_local[:, 2, 3].float().cpu()
  g = g_local.cpu()
  gx, gz = float(g[0, 3].item()), float(g[2, 3].item())

  x_lo, x_hi, z_lo, z_hi = _percentile_bounds(px, pz, gx, gz, bounds_pct, bounds_pad)

  if px.numel() > 0:
    ax.scatter(px.numpy(), pz.numpy(), s=8, alpha=0.2, color='crimson')
  ax.scatter([gx], [gz], s=40, color='black', marker='o')
  ax.scatter([0.0], [0.0], s=40, color='royalblue', marker='o')

  ax.set_xlim(x_lo, x_hi)
  ax.set_ylim(z_lo, z_hi)
  ax.set_aspect('equal')
  ax.grid(True)
  ax.set_xlabel('dx (m, local frame)')
  ax.set_ylabel('dz (m, local frame)')
  ax.set_title(f'x-z · step {step_idx + 1}/{n_steps}')


def _legend_handles(n_particles: int) -> list[Line2D]:
  return [
    Line2D([], [], marker='^', linestyle='', color='royalblue', label='origin'),
    Line2D([], [], marker='^', linestyle='', color='crimson', alpha=0.6, label=f'particles (N={n_particles})'),
    Line2D([], [], marker='^', linestyle='', color='black', label='ground truth'),
  ]


def _build_motion_figure(se3: bool, n_particles: int):
  if se3:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax_xy, ax_xz = axes[0], axes[1]
  else:
    fig, ax_xy = plt.subplots(figsize=(5, 5))
    ax_xz = None
  fig.subplots_adjust(top=0.82)
  fig.legend(
    handles=_legend_handles(n_particles),
    loc='upper center',
    bbox_to_anchor=(0.5, 0.98),
    ncols=3,
  )
  return fig, ax_xy, ax_xz


def render_motion(
    snapshots: torch.Tensor,
    x_ref: torch.Tensor,
    output_stem: Path,
    fps: int = 4,
    bounds_pct: float = 98.0,
    bounds_pad: float = 0.10,
    frame_index: int = -1,
    se3: bool = False,
) -> tuple[Path, Path]:
  """Render motion-model video (+ representative PNG) for one sequence.

  Args:
    snapshots: (N_steps, N, 8) UDQ particles after each motion step.
    x_ref: (N_steps + 1, 8) UDQ ground-truth poses for the sequence.
    output_stem: base path without extension.
    se3: when True, also render a side x-z subplot.

  Returns:
    (video_path, png_path)
  """
  n_steps = int(snapshots.shape[0])
  if x_ref.shape[0] != n_steps + 1:
    raise ValueError(
      f'x_ref must have N_steps+1 poses, got {x_ref.shape[0]} for N_steps={n_steps}'
    )
  if snapshots.shape[-1] != 8 or x_ref.shape[-1] != 8:
    raise ValueError(
      f'render_motion expects UDQ inputs (...,8); got snapshots={tuple(snapshots.shape)} '
      f'x_ref={tuple(x_ref.shape)}'
    )

  # Local-frame UDQs, then a one-shot conversion to 4x4 for matplotlib drawing.
  ref_inv = udq_conj(x_ref[:-1])                                   # (T-1, 8)
  p_local_udq = udq_mul(ref_inv.unsqueeze(1), snapshots)           # (T-1, N, 8)
  g_local_udq = udq_mul(ref_inv, x_ref[1:])                        # (T-1, 8)
  p_local = udq_to_matrix(p_local_udq)                             # (T-1, N, 4, 4)
  g_local = udq_to_matrix(g_local_udq)                             # (T-1, 4, 4)

  n_particles = int(snapshots.shape[1])

  fig, ax_xy, ax_xz = _build_motion_figure(se3, n_particles)

  def update(t: int):
    _render_xy(ax_xy, p_local[t], g_local[t], bounds_pct, bounds_pad, t, n_steps)
    if ax_xz is not None:
      _render_xz(ax_xz, p_local[t], g_local[t], bounds_pct, bounds_pad, t, n_steps)
    return []

  anim = FuncAnimation(fig, update, frames=n_steps, interval=1000 // max(fps, 1), blit=False)

  video_path = output_stem.with_suffix('.mp4')
  video_path.parent.mkdir(parents=True, exist_ok=True)
  try:
    anim.save(str(video_path), writer=FFMpegWriter(fps=fps))
  except (FileNotFoundError, RuntimeError):
    video_path = output_stem.with_suffix('.gif')
    anim.save(str(video_path), writer=PillowWriter(fps=fps))

  plt.close(fig)

  # Representative single frame.
  t_single = (n_steps - 1) // 2 if frame_index < 0 else max(0, min(frame_index, n_steps - 1))

  fig, ax_xy, ax_xz = _build_motion_figure(se3, n_particles)
  _render_xy(ax_xy, p_local[t_single], g_local[t_single], bounds_pct, bounds_pad, t_single, n_steps)
  if ax_xz is not None:
    _render_xz(ax_xz, p_local[t_single], g_local[t_single], bounds_pct, bounds_pad, t_single, n_steps)
  png_path = output_stem.parent / f'{output_stem.name}_t{t_single}.png'
  fig.savefig(png_path, dpi=150, bbox_inches='tight')
  plt.close(fig)

  return video_path, png_path


def _build_sequence_dataset(
    dataset: Path,
    lidar_topic: str,
    odometry_topic: str,
    steps: int,
) -> SequenceDataset:
  base = LocalisationDataset(dataset, lidar_topic=lidar_topic, odometry_topic=odometry_topic)
  return SequenceDataset(base, step_count=steps, downsample=0)


def _render_motion_for_model(
    motion_model: nn.Module,
    is_se3: bool,
    label: str,
    ds: SequenceDataset,
    indices: list[int],
    artifact_dir: Path,
    particles: int,
    fps: int,
    bounds_pct: float,
    bounds_pad: float,
    frame_index: int,
    device: torch.device,
) -> tuple[list[str], list[str]]:
  """Render motion snapshots for every selected sequence; returns (video_paths, frame_paths)."""
  video_paths: list[str] = []
  frame_paths: list[str] = []
  with torch.no_grad():
    for idx in tqdm(indices, desc=f'motion eval {label}'):
      _, _, x_ref, odometry = ds[idx]
      snapshots = motion_snapshots_udq(
        motion_model,
        particles,
        x_ref.to(device),
        odometry.to(device),
      )
      stem = artifact_dir / f'trajectory_seq-{idx}'
      video_path, png_path = render_motion(
        snapshots=snapshots.cpu(),
        x_ref=x_ref.cpu(),
        output_stem=stem,
        fps=fps,
        bounds_pct=bounds_pct,
        bounds_pad=bounds_pad,
        frame_index=frame_index,
        se3=is_se3,
      )
      video_paths.append(str(video_path))
      frame_paths.append(str(png_path))
  return video_paths, frame_paths


def evaluate_motion(
    checkpoints: list[Path],
    dataset: Path,
    lidar_topic: str,
    odometry_topic: str,
    steps: int,
    sequences: int,
    particles: int,
    fps: int,
    bounds_pct: float,
    bounds_pad: float,
    frame_index: int,
    device: torch.device,
    run_dir: Path,
) -> list[dict[str, object]]:
  if not checkpoints:
    return []

  ds = _build_sequence_dataset(dataset, lidar_topic, odometry_topic, steps)
  indices = select_indices(len(ds), sequences)
  results: list[dict[str, object]] = []

  for checkpoint in checkpoints:
    print(f'[evaluate_motion] checkpoint={checkpoint}')
    cfg, loaded = ModuleConfig.load_module(checkpoint)
    if isinstance(cfg, DPFConfig):
      motion_model = loaded.dynamics
    elif isinstance(cfg, DynamicsConfig):
      motion_model = loaded
    else:
      print(f'[evaluate_motion] skip {type(cfg).__name__}: not a motion or DPF checkpoint')
      continue
    motion_model.to(device).eval()

    assert isinstance(motion_model, UDQ_MOTION_TYPES), (
      f'Unsupported motion-model type for UDQ evaluator: {type(motion_model).__name__}'
    )
    is_se3 = isinstance(motion_model, SE3_MOTION_TYPES)

    artifact_dir = _create_checkpoint_artifact_dir(run_dir, checkpoint)
    symlink_path = _create_model_symlink(artifact_dir, checkpoint)

    video_paths, frame_paths = _render_motion_for_model(
      motion_model=motion_model,
      is_se3=is_se3,
      label=checkpoint.name,
      ds=ds,
      indices=indices,
      artifact_dir=artifact_dir,
      particles=particles,
      fps=fps,
      bounds_pct=bounds_pct,
      bounds_pad=bounds_pad,
      frame_index=frame_index,
      device=device,
    )

    results.append({
      'checkpoint': str(checkpoint),
      'dataset': str(dataset),
      'sequences': len(indices),
      'videos': ';'.join(video_paths),
      'frames': ';'.join(frame_paths),
      'artifact_dir': str(artifact_dir),
      'model_symlink': symlink_path,
    })

  return results


def _build_baseline(name: str, std_t: float, std_r: float) -> tuple[nn.Module, bool]:
  if name == 'odometry-se3':
    return OdometryDynamicsModelSE3(std_t=std_t, std_r=std_r), True
  raise ValueError(f'Unknown baseline {name!r}; choose from {BASELINE_CHOICES}')


def evaluate_baselines(
    baselines: list[str],
    dataset: Path,
    lidar_topic: str,
    odometry_topic: str,
    steps: int,
    sequences: int,
    particles: int,
    fps: int,
    bounds_pct: float,
    bounds_pad: float,
    frame_index: int,
    std_t: float,
    std_r: float,
    device: torch.device,
    run_dir: Path,
) -> list[dict[str, object]]:
  if not baselines:
    return []

  ds = _build_sequence_dataset(dataset, lidar_topic, odometry_topic, steps)
  indices = select_indices(len(ds), sequences)
  results: list[dict[str, object]] = []

  for name in baselines:
    motion_model, is_se3 = _build_baseline(name, std_t=std_t, std_r=std_r)
    motion_model.to(device)
    motion_model.eval()
    print(
      f'[evaluate_baselines] baseline={name} std_t={std_t:.4g} std_r={std_r:.4g} se3={is_se3}'
    )

    artifact_dir = run_dir / f'baseline__{name}'
    artifact_dir.mkdir(parents=True, exist_ok=True)

    video_paths, frame_paths = _render_motion_for_model(
      motion_model=motion_model,
      is_se3=is_se3,
      label=name,
      ds=ds,
      indices=indices,
      artifact_dir=artifact_dir,
      particles=particles,
      fps=fps,
      bounds_pct=bounds_pct,
      bounds_pad=bounds_pad,
      frame_index=frame_index,
      device=device,
    )

    results.append({
      'checkpoint': f'baseline:{name}',
      'dataset': str(dataset),
      'sequences': len(indices),
      'videos': ';'.join(video_paths),
      'frames': ';'.join(frame_paths),
      'artifact_dir': str(artifact_dir),
      'model_symlink': '',
    })

  return results


def print_motion_results(results: list[dict[str, object]]):
  if not results:
    print('No motion checkpoints found or evaluated.')
    return

  print('\n== Motion Model Analysis (per-checkpoint artifacts; no scalar metric) ==')
  print(f"{'rank':>4}  {'seq':>4}  {'dataset':<24}  {'checkpoint':<48}  artifact_dir")
  for rank, row in enumerate(results, start=1):
    sequences = cast(int, row['sequences'])
    dataset = cast(str, row.get('dataset', '-'))
    checkpoint = cast(str, row['checkpoint'])
    artifact_dir = cast(str, row['artifact_dir'])
    print(f"{rank:4d}  {sequences:4d}  {dataset:<24}  {checkpoint:<48}  {artifact_dir}")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description='Render motion-model snapshots from DPF checkpoints.')

  parser.add_argument('--dataset', nargs='+', type=Path, default=DEFAULT_EVAL_DATASETS, help='Path(s) to ROS2 bag folders.')
  parser.add_argument('--models-dir', type=Path, default=Path('models'), help='Directory containing the dpf/ checkpoint folder.')
  parser.add_argument('--lidar-topic', default='/livox/lidar', help='LIDAR topic inside the ROS2 bag.')
  parser.add_argument('--odometry-topic', default='/odometry/filtered', help='Odometry topic inside the ROS2 bag.')
  parser.add_argument('--max-checkpoints', type=int, default=0, help='Max checkpoints to evaluate (0 evaluates all).')
  parser.add_argument('--output-root', type=Path, default=Path('eval_runs'), help='Directory where timestamped evaluation folders are created.')

  parser.add_argument('--sequences', type=int, default=3, help='Replay sequences to render per checkpoint.')
  parser.add_argument('--steps', type=int, default=10, help='Replay sequence length (number of poses; snapshot frames = steps - 1).')
  parser.add_argument('--particles', type=int, default=350, help='Particle count for motion-model spread.')
  parser.add_argument('--frame-index', type=int, default=-1, help='Zero-based step index for the single-frame PNG (-1 selects the middle step).')
  parser.add_argument('--fps', type=int, default=4, help='Video frame rate.')
  parser.add_argument('--bounds-pct', type=float, default=98.0, help='Percentile of |dx|,|dy| across a video used to set axis bounds.')
  parser.add_argument('--bounds-pad', type=float, default=0.10, help='Extra padding factor on top of the percentile bound.')

  parser.add_argument(
    '--baseline',
    nargs='+',
    choices=BASELINE_CHOICES,
    default=None,
    help='Evaluate parameterless odometry baseline(s) instead of checkpoints. Skips ckpt discovery.',
  )
  parser.add_argument('--baseline-std-t', type=float, default=0.1, help='Translation std for the odometry baseline (scalar; isotropic on x/y).')
  parser.add_argument('--baseline-std-r', type=float, default=0.05, help='Rotation std for the odometry baseline (scalar; isotropic).')

  return parser


def main():
  args = build_parser().parse_args()

  for ds_path in args.dataset:
    if not ds_path.exists():
      raise FileNotFoundError(f'Dataset path does not exist: {ds_path}')

  device = get_device()
  run_dir = _make_run_dir(args.output_root)

  if args.baseline:
    _save_run_metadata(run_dir, args, device, [])
    all_results: list[dict[str, object]] = []
    for ds_path in args.dataset:
      results = evaluate_baselines(
        baselines=list(args.baseline),
        dataset=ds_path,
        lidar_topic=args.lidar_topic,
        odometry_topic=args.odometry_topic,
        steps=args.steps,
        sequences=args.sequences,
        particles=args.particles,
        fps=args.fps,
        bounds_pct=args.bounds_pct,
        bounds_pad=args.bounds_pad,
        frame_index=args.frame_index,
        std_t=args.baseline_std_t,
        std_r=args.baseline_std_r,
        device=device,
        run_dir=run_dir,
      )
      all_results.extend(results)
    print_motion_results(all_results)
    return

  checkpoints = discover_motion_checkpoints(args.models_dir, max_checkpoints=args.max_checkpoints)
  if not checkpoints:
    print(f'No motion checkpoints found under {args.models_dir.resolve()}')
    return

  _save_run_metadata(run_dir, args, device, checkpoints)

  all_results = []
  for ds_path in args.dataset:
    results = evaluate_motion(
      checkpoints=checkpoints,
      dataset=ds_path,
      lidar_topic=args.lidar_topic,
      odometry_topic=args.odometry_topic,
      steps=args.steps,
      sequences=args.sequences,
      particles=args.particles,
      fps=args.fps,
      bounds_pct=args.bounds_pct,
      bounds_pad=args.bounds_pad,
      frame_index=args.frame_index,
      device=device,
      run_dir=run_dir,
    )
    all_results.extend(results)

  print_motion_results(all_results)


if __name__ == '__main__':
  main()
