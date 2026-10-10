import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import numpy
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector
from open3d.visualization import Visualizer, draw_geometries
from plotly import graph_objects as go
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .bag_parser import read_bag, parse_lidar, LocalisationDataset
from .dpf import SequenceDataset, DPFConfig, init_se2_like_particles
from .obs import FixedPillarEncoder, AlignmentDataset, MultiSDF
from .pose import udq_mul, udq_conj, udq_from_qt, udq_to_matrix, udq_transform_points_fast
from .util import get_device, sample_frame

def get_world_map_3d(ds: LocalisationDataset, points: int = 5000):
  return ds.world[::max(1, (len(ds.world) // points))].cpu().numpy()[:points]
  
def get_sdf_map_3d(sdf: MultiSDF, gridres: int = 200):
  offset = sdf.offset.cpu()
  scale = sdf.scale.cpu()
  
  x = torch.linspace(offset[0] - scale[0], offset[0] + scale[0], gridres)
  y = torch.linspace(offset[1] - scale[1], offset[1] + scale[1], gridres)
  z = torch.linspace(offset[2] - scale[2], offset[2] + scale[2], gridres)
  grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
  
  points = torch.stack([grid_x, grid_y, grid_z], dim=-1)
  points_flat = points.view(-1, 3)
  
  # Optimization: Evaluate SDF in chunks to avoid memory spikes from large MLP activations
  dist_list = []
  chunk_size = 250000
  with torch.no_grad():
      for i in range(0, points_flat.shape[0], chunk_size):
          chunk = points_flat[i : i + chunk_size].to(get_device())
          dist_chunk = sdf(chunk).cpu()
          dist_list.append(dist_chunk)
  
  dist = torch.cat(dist_list, dim=0).view(gridres, gridres, gridres)
  
  mask = torch.abs(dist) < 0.05
  return points[mask].numpy()

def get_pcd(points: torch.Tensor, color: Optional[torch.Tensor] = None) -> PointCloud:
  pcd = PointCloud()
  pcd.points = Vector3dVector(points[..., :3].cpu())

  if color is not None:
    pcd.colors = Vector3dVector(torch.ones_like(points[..., :3].cpu()) * color)

  return pcd


def show_pcds(frames: Iterable[torch.Tensor]):
  vis = Visualizer()
  vis.create_window(width=3000, height=1500)

  pcd = None

  for f in frames:
    if pcd is None:
      pcd = get_pcd(f)
      vis.add_geometry(pcd)
    else:
      pcd.points = Vector3dVector(f[..., :3].cpu())
      vis.update_geometry(pcd)
      vis.poll_events()
      vis.update_renderer()

  vis.destroy_window()


# render lidar from a ROS2 bag
def show_bag(path: Path, lidar_topic: str = '/livox/lidar'):
  frames = read_bag(path, (lidar_topic,))
  show_pcds(parse_lidar(msg) for _, _, msg in frames)


# show the SLAM global map
def show_slam(path: Path):
  draw_geometries([get_pcd(LocalisationDataset(path).world)])


# show the world map from an np array
def show_world_np(world_np):
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111, projection='3d')
  
  ax.scatter(world_np[:, 0], world_np[:, 1], world_np[:, 2], s=0.1, c=world_np[:, 2], cmap='viridis', alpha=0.5)
  
  # Set equal aspect ratio
  mid_x = (world_np[:, 0].max() + world_np[:, 0].min()) * 0.5
  mid_y = (world_np[:, 1].max() + world_np[:, 1].min()) * 0.5
  mid_z = (world_np[:, 2].max() + world_np[:, 2].min()) * 0.5
  max_range = numpy.array([world_np[:, 0].max()-world_np[:, 0].min(), 
                          world_np[:, 1].max()-world_np[:, 1].min(), 
                          world_np[:, 2].max()-world_np[:, 2].min()]).max() * 0.5
  ax.set_xlim(mid_x - max_range, mid_x + max_range)
  ax.set_ylim(mid_y - max_range, mid_y + max_range)
  ax.set_zlim(mid_z - max_range, mid_z + max_range)

  ax.set_title(f"World Map (Showing {len(world_np)} points)")
  ax.set_xlabel("X (m)")
  ax.set_ylabel("Y (m)")
  ax.set_zlabel("Z (m)")

  plt.tight_layout()
  plt.show()

def show_sdf_np(sdf_np):
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111, projection='3d')

  if sdf_np.shape[0] > 0:
      ax.scatter(sdf_np[:, 0], sdf_np[:, 1], sdf_np[:, 2], s=1, c=sdf_np[:, 2], cmap='viridis')
  else:
      raise ValueError("No surface points found with dist < 0.05")
  
  # Set equal aspect ratio
  mid_x = (sdf_np[:, 0].max() + sdf_np[:, 0].min()) * 0.5
  mid_y = (sdf_np[:, 1].max() + sdf_np[:, 1].min()) * 0.5
  mid_z = (sdf_np[:, 2].max() + sdf_np[:, 2].min()) * 0.5
  max_range = numpy.array([sdf_np[:, 0].max()-sdf_np[:, 0].min(), 
                           sdf_np[:, 1].max()-sdf_np[:, 1].min(), 
                           sdf_np[:, 2].max()-sdf_np[:, 2].min()]).max() * 0.5

  ax.set_xlim(mid_x - max_range, mid_x + max_range)
  ax.set_ylim(mid_y - max_range, mid_y + max_range)
  ax.set_zlim(mid_z - max_range, mid_z + max_range)

  ax.set_title(f"SDF (showing {len(sdf_np)} voxels)")
  ax.set_xlabel("X (m)")
  ax.set_ylabel("Y (m)")
  ax.set_zlabel("Z (m)")

  plt.tight_layout()
  plt.show()


# show channels from the voxel encoder output
def show_bev(path: Path):
  data = LocalisationDataset(path)
  encoder = FixedPillarEncoder(torch.tensor([350, 350]), 0.15, 0.154)

  world = encoder(data.world).detach().cpu()

  plt.subplot(131)
  plt.imshow(world[1] - world[2] * 2, cmap='inferno')
  plt.subplot(132)
  plt.imshow(world[3], cmap='inferno')
  plt.subplot(133)
  plt.imshow(world[4], cmap='inferno')
  plt.gcf().set_size_inches(12, 4)
  plt.tight_layout()
  plt.show()


# plot the observation model likelihood score
def show_score(path: Path, observer, idx: int = 20, N: int = 31, save_filepath: Optional[Path] = None):
  device = get_device()
  ds = AlignmentDataset(path, map_device=device, downsample_world=150000)

  observer = observer.to(device).eval()

  W1, W2 = [], []

  ts = torch.linspace(-1, 1, N, device=device)
  u, v = torch.cartesian_prod(ts, ts).unbind(-1)

  # offsets1: translation (u, v), no rotation
  q1 = torch.tensor([1., 0., 0., 0.], device=device).expand(u.shape[0], -1)
  t1 = torch.stack([u, v, torch.zeros_like(u)], dim=-1)
  offsets1 = udq_from_qt(q1, t1)

  # offsets2: translation (v, 0), rotation u (heading)
  q2 = torch.stack([torch.cos(u / 2), torch.zeros_like(u), torch.zeros_like(u), torch.sin(u / 2)], dim=-1)
  t2 = torch.stack([v, torch.zeros_like(v), torch.zeros_like(v)], dim=-1)
  offsets2 = udq_from_qt(q2, t2)

  pose = ds.poses[idx].to(device)

  with torch.no_grad():
    for o1, o2 in tqdm(DataLoader(TensorDataset(offsets1, offsets2), batch_size=64, shuffle=False), desc='score'):
      w, s = ds.get_perturbed(ds.scans[idx].to(device), pose, o1)
      W1.append(observer(s, w).detach().cpu())

      w, s = ds.get_perturbed(ds.scans[idx].to(device), pose, o2)
      W2.append(observer(s, w).detach().cpu())

  W1 = torch.cat(W1).reshape(N, N)
  W2 = torch.cat(W2).reshape(N, N)

  fig = plt.figure(figsize=(10, 5))
  plt.subplot(121)
  plt.imshow(W2, extent=(-1, 1, -1, 1))
  plt.axvline(0, linestyle=':', linewidth=1, color='white')
  plt.axhline(0, linestyle=':', linewidth=1, color='white')
  plt.xlabel('x offset (m)')
  plt.ylabel('heading offset (rad)')
  plt.subplot(122)
  plt.imshow(W1, extent=(-1, 1, -1, 1))
  plt.axvline(0, linestyle=':', linewidth=1, color='white')
  plt.axhline(0, linestyle=':', linewidth=1, color='white')
  plt.xlabel('x offset (m)')
  plt.ylabel('y offset (m)')
  plt.tight_layout()

  if save_filepath is not None:
      plt.savefig(save_filepath, dpi=300, bbox_inches='tight')
  else:
      plt.show()

  return fig


# plot the reference and estimated trajectory
def show_trajectory(path: Path, checkpoint: Path, downsample: int = 8000, steps: int = 10, idx: int = 27):
  device = get_device()

  _, model = DPFConfig.load_module(checkpoint)
  model.to(device).eval()

  base = LocalisationDataset(path)
  ds = SequenceDataset(base, step_count=steps)

  world, scans, x_ref, odometry = (t.to(device) for t in ds[idx])
  if downsample > 0:
    scans = torch.stack([sample_frame(s, downsample) for s in scans])

  x0 = x_ref[0]
  p0 = udq_mul(init_se2_like_particles(model.particle_count).to(device), x0)
  u = udq_mul(udq_conj(odometry[:-1]), odometry[1:])

  with torch.no_grad():
    x_out = model(scans, x0, p0, u, world)
  x_out = torch.cat([x_ref[:1], x_out])

  # boundary: matplotlib needs 4x4 matrices for 2D pose drawing
  x_ref_m = udq_to_matrix(x_ref).cpu()
  x_out_m = udq_to_matrix(x_out).cpu()

  t_err = torch.norm(x_out_m[1:, :2, 3] - x_ref_m[1:, :2, 3], dim=-1)
  print(f't err: {torch.max(t_err):.3f} (max) / {torch.std(t_err):.3f} (std)')

  a_err = torch.abs(torch.atan2(x_out_m[1:, 1, 0], x_out_m[1:, 0, 0]) - torch.atan2(x_ref_m[1:, 1, 0], x_ref_m[1:, 0, 0]))
  print(f'a err: {torch.max(a_err):.3f} (max) / {torch.std(a_err):.3f} (std)')

  plt.style.use('ggplot')

  triangle = torch.tensor([
    [0.30, 0.00],
    [-0.20, 0.15],
    [-0.20, -0.15]
  ]) * 0.15

  pc = PatchCollection([Polygon(triangle @ x[:2, :2].T + x[:2, 3], closed=True) for x in x_ref_m])
  pc.set_color('black')
  plt.gca().add_collection(pc)

  pc = PatchCollection([Polygon(triangle @ x[:2, :2].T + x[:2, 3], closed=True) for x in x_out_m])
  pc.set_color('crimson')
  plt.gca().add_collection(pc)

  plt.plot(x_ref_m[:, 0, 3], x_ref_m[:, 1, 3], '--', c='black', label='reference')
  plt.plot(x_out_m[:, 0, 3], x_out_m[:, 1, 3], '--', c='crimson', label='estimator')

  plt.legend()
  plt.xlabel('x position (m)')
  plt.ylabel('y position (m)')
  plt.gca().set_aspect('equal')
  plt.tight_layout()
  plt.show()


@dataclass
class PoseSet:
  q: torch.Tensor
  v: torch.Tensor = None
  name: str = None
  marker: str = None
  connected: bool = False


def show_se3_poses(*pose_sets: PoseSet, title: str | None = None, scale: float = 0.2, save_tag: str | None = None):
  fig = go.Figure()

  for i, poses in enumerate(pose_sets):
    name = poses.name or f'pose set {i}'

    with torch.no_grad():
      M = udq_to_matrix(poses.q)
      t = M[:, :3, 3].cpu()
      R = M[:, :3, :3].cpu()

    if poses.connected:
      color = poses.connected if isinstance(poses.connected, str) else 'gray'

      fig.add_trace(go.Scatter3d(
        x=t[:, 0], y=t[:, 1], z=t[:, 2],
        mode='lines',
        line={'color': color, 'width': 5},
        name=f'{name} trajectory',
        opacity=0.6,
        hoverinfo='none'
      ))

    fig.add_trace(go.Scatter3d(
      x=t[:, 0], y=t[:, 1], z=t[:, 2],
      mode='markers',
      marker={
        'size': 7,
        'symbol': poses.marker or 'circle',
        'line': {'width': 1, 'color': 'gray'},
        'color': 'gray' if poses.v is None else poses.v
      },
      name=f'{name} origins'
    ))

    for j, (axis_color, axis_name) in enumerate([('red', 'x'), ('green', 'y'), ('blue', 'z')]):
      lines = torch.full((3 * t.shape[0], 3), torch.nan)
      lines[0::3] = t
      lines[1::3] = t + R[:, :, j] * scale

      fig.add_trace(go.Scatter3d(
        x=lines[:, 0],
        y=lines[:, 1],
        z=lines[:, 2],
        mode='lines',
        line={'color': axis_color, 'width': 4},
        name=f'{name} {axis_name}-axis',
        opacity=0.6,
        hoverinfo='none'
      ))

  fig.update_layout(
    title=title,
    scene={'aspectmode': 'data', 'xaxis_title': 'x', 'yaxis_title': 'y', 'zaxis_title': 'z'},
  )

  if save_tag is None:
    os.makedirs('../cache/show_se3_poses', exist_ok=True)
    fig.write_html('../cache/show_se3_poses/out.html', auto_open=True, include_plotlyjs='cdn')
  else:
    path = Path(save_tag).resolve()
    fig.write_html(path, auto_open=False)


def plot_obs_landscape(obs: nn.Module, scan: torch.Tensor, gt_pose: torch.Tensor, device: torch.device, save_path: str | Path, steps: int = 21, use_energy: bool = True):
  dx = torch.linspace(-2, 2, steps)
  dy = torch.linspace(-2, 2, steps)
  dr = torch.linspace(-torch.pi / 3, torch.pi / 3, steps)

  out_xy = torch.zeros(steps, steps)
  out_xr = torch.zeros(steps, steps)

  obs.eval()
  scan = scan.to(device).unsqueeze(0)
  gt_pose = gt_pose.to(device).unsqueeze(0)

  with torch.no_grad():
    for i, x in enumerate(dx):
      for j, y in enumerate(dy):
        offset = udq_from_qt(
          q=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
          t=torch.tensor([[x.item(), y.item(), 0.0]], device=device)
        )

        points = udq_transform_points_fast(udq_mul(gt_pose, offset), scan)
        out_xy[j, i] = obs.get_energy(points) if use_energy else obs(points)

    for i, x in enumerate(dx):
      for j, ang in enumerate(dr):
        offset = udq_from_qt(
          q=torch.tensor([[math.cos(ang.item() / 2), 0.0, 0.0, math.sin(ang.item() / 2)]], device=device),
          t=torch.tensor([[x.item(), 0.0, 0.0]], device=device)
        )

        points = udq_transform_points_fast(udq_mul(gt_pose, offset), scan)
        out_xr[j, i] = obs.get_energy(points) if use_energy else obs(points)

  fig, ax = plt.subplots(1, 2, figsize=(14, 6))

  c1 = ax[0].pcolormesh(dx.numpy(), dy.numpy(), out_xy)
  ax[0].set_title(r'output for xy-translation offset')
  ax[0].set_xlabel('local x offset (m)')
  ax[0].set_ylabel('local y offset (m)')
  ax[0].axhline(0, color='white', linestyle='--', alpha=0.5)
  ax[0].axvline(0, color='white', linestyle='--', alpha=0.5)
  fig.colorbar(c1, ax=ax[0])

  c2 = ax[1].pcolormesh(dx.numpy(), dr.numpy(), out_xr)
  ax[1].set_title(r'output for translation/rotation offset')
  ax[1].set_xlabel('local x offset (m)')
  ax[1].set_ylabel('local z rotation (rad)')
  ax[1].axhline(0, color='white', linestyle='--', alpha=0.5)
  ax[1].axvline(0, color='white', linestyle='--', alpha=0.5)
  fig.colorbar(c2, ax=ax[1])

  plt.tight_layout()
  plt.savefig(save_path, dpi=150)
  plt.close(fig)
