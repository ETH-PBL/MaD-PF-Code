import typing
from collections import defaultdict, deque
import math
from operator import itemgetter
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from pypcd4 import PointCloud
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore, Stores
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from .pose import udq_conj, udq_from_qt, udq_mul, udq_transform_points
from .slam import run_slam, register_clouds
from .util import disk_cache

# Rig-specific calibrations missing from recorded /tf*. Frame names normalized (no leading '/').
# jackal/etf/street: livox sits in place of ouster, flipped 180° in yaw → os_lidar → livox_frame
_IDENTITY_UDQ = udq_from_qt(torch.tensor([1., 0, 0, 0]), torch.zeros(3))
_RZ180_UDQ = udq_from_qt(
  torch.tensor([math.cos(math.pi / 2), 0., 0., math.sin(math.pi / 2)], dtype=torch.float32),
  torch.zeros(3, dtype=torch.float32),
)
KNOWN_STATIC_TF: dict[tuple[str, str], torch.Tensor] = {
  # TODO: compute correct static TF
  # jackal base→mount chain (URDF nominals; missing from some etf_floorF bags, e.g. f9)
  ('base_link', 'chassis_link'): _IDENTITY_UDQ.clone(),
  ('chassis_link', 'mid_mount'): udq_from_qt(
    torch.tensor([1., 0, 0, 0], dtype=torch.float32),
    torch.tensor([0., 0., 0.092], dtype=torch.float32),
  ),
  ('mid_mount', 'rear_mount'): udq_from_qt(
    torch.tensor([1., 0, 0, 0], dtype=torch.float32),
    torch.tensor([-0.06, 0., 0.], dtype=torch.float32),
  ),
  # ouster mount on jackal rig (some bags only publish this on /tf, some omit it entirely)
  ('rear_mount', 'os_sensor'): udq_from_qt(
    torch.tensor([1., 0, 0, 0], dtype=torch.float32),
    torch.tensor([-0.040, 0., 0.088], dtype=torch.float32),
  ),
  ('os_sensor', 'os_lidar'): udq_from_qt(
    torch.tensor([1., 0, 0, 0], dtype=torch.float32),
    torch.tensor([0., 0., 0.036], dtype=torch.float32),
  ),
  ('os_lidar', 'livox_frame'): _RZ180_UDQ,
  # Calibrated base_link -> livox_frame for ethmocap dataset (with 180 deg Z swap)
  ('base_link', 'livox_frame'): udq_from_qt(
    torch.tensor([0.7102, 0.0017, -0.0063, -0.7040], dtype=torch.float32),
    torch.tensor([-0.0094, -0.1275, 0.1049], dtype=torch.float32),
  ),
}


def read_bag(path: Path, topics: tuple[str | None, ...]) -> Iterable[tuple[str, float, object]]:
  ts0 = None

  with AnyReader([path], default_typestore=get_typestore(Stores.ROS2_KILTED)) as reader:
    for connection, ts, raw in reader.messages():

      if connection.topic in topics:
        msg = reader.deserialize(raw, connection.msgtype)

        if ts0 is None:
          ts0 = msg.header.stamp.sec

        yield connection.topic, msg.header.stamp.sec - ts0 + msg.header.stamp.nanosec * 1e-9, msg


def read_bag_sync(path: Path, topics: tuple[str | None, ...]) -> Iterable[tuple[object]]:
  times, messages = defaultdict(list), defaultdict(list)

  # check topics exist
  with AnyReader([path], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
    existing_topics = {c.topic for c in reader.connections}
    for topic in topics:
      if topic not in existing_topics:
        raise ValueError(f'Topic {topic!r} not found in bag; available topics: {existing_topics}')

  for topic, ts, msg in read_bag(path, topics):
    times[topic].append(ts), messages[topic].append(msg)

  ts0 = torch.tensor(times[topics[0]]).unsqueeze(-1)
  return (itemgetter(*(ts0 - torch.tensor(times[topic])).abs().argmin(dim=-1))(messages[topic]) for topic in topics)


def parse_lidar(msg: object) -> torch.Tensor:
  return torch.tensor(msg.data.reshape((-1, msg.point_step))[:, :16].view('<f4').copy())


def parse_pose(msg: object) -> torch.Tensor:
  while hasattr(msg, 'pose'):
    msg = msg.pose  # unpack nested

  q = msg.orientation
  t = msg.position

  return udq_from_qt(
    torch.tensor([q.w, q.x, q.y, q.z], dtype=torch.float32),
    torch.tensor([t.x, t.y, t.z], dtype=torch.float32),
  )


def mask_robot(frame: torch.Tensor, size: float = 0.5) -> torch.Tensor:
  return frame[(frame[..., 0] > size) | (frame[..., 1] > size) | (frame[..., 0] < -size) | (frame[..., 1] < -size)]


def tf_to_udq(node) -> torch.Tensor:
  return udq_from_qt(
    torch.from_numpy(np.roll(Rotation.from_matrix(node.pose[..., :3, :3]).as_quat(), 1)),
    torch.from_numpy(node.pose[..., :3, 3].copy())
  ).to(torch.float32)

def extract_mocap_gt(
  data_path: Path,
  mocap_path: Path,
  lidar_topic: str = '/livox/lidar',
  pose_topic: str = '/Jackal/pose',
  base_to_lidar: Optional[torch.Tensor] = None,
  swap_forward: bool = False
) -> torch.Tensor:
  """Sync mocap poses from data_path/gt/ bag to lidar frame timestamps and transform to lidar frame.

  :param base_to_lidar: (8,) UDQ or (4, 4) matrix representing T_base_lidar. 
                        If provided, transforms mocap (base) poses to lidar poses: T_w_l = T_w_b * T_b_l
  :param swap_forward: If True, rotates the mocap base frame by 180 degrees around Z axis 
                       to correct for forward/backward swap.
  :return: (N, 8) UDQ tensor, one mocap pose in lidar frame per lidar frame
  """
  typestore = get_typestore(Stores.ROS2_KILTED)

  lidar_ts: list[int] = []
  with AnyReader([data_path], default_typestore=typestore) as reader:
    conns = [c for c in reader.connections if c.topic == lidar_topic]
    for _, ts_ns, _ in reader.messages(connections=conns):
      lidar_ts.append(ts_ns)

  mocap_ts: list[int] = []
  mocap_poses: list[torch.Tensor] = []
  with AnyReader([mocap_path], default_typestore=typestore) as reader:
    conns = [c for c in reader.connections if c.topic == pose_topic]
    for conn, ts_ns, raw in reader.messages(connections=conns):
      mocap_ts.append(ts_ns)
      mocap_poses.append(parse_pose(reader.deserialize(raw, conn.msgtype)))

  lidar_t = torch.tensor(lidar_ts, dtype=torch.int64).unsqueeze(-1)
  mocap_t = torch.tensor(mocap_ts, dtype=torch.int64)
  idx = torch.argmin(torch.abs(lidar_t - mocap_t), dim=-1)

  poses = torch.stack([mocap_poses[i] for i in idx])

  if swap_forward:
    # Rotate 180 deg around Z: q = [cos(pi/2), 0, 0, sin(pi/2)] = [0, 0, 0, 1]
    rz180 = udq_from_qt(torch.tensor([0., 0., 0., 1.]), torch.zeros(3))
    poses = udq_mul(poses, rz180)

  if base_to_lidar is not None:
    if base_to_lidar.shape == (4, 4):
      q = torch.from_numpy(np.roll(Rotation.from_matrix(base_to_lidar[:3, :3].numpy()).as_quat(), 1)).to(torch.float32)
      t = base_to_lidar[:3, 3].to(torch.float32)
      base_to_lidar = udq_from_qt(q, t)
    
    # T_world_lidar = T_world_base * T_base_lidar
    poses = udq_mul(poses, base_to_lidar)

  return poses


def _norm_frame(f: str) -> str:
  return f.lstrip('/')


def _tf_msg_to_udq(transform) -> torch.Tensor:
  q, t = transform.rotation, transform.translation
  return udq_from_qt(
    torch.tensor([q.w, q.x, q.y, q.z], dtype=torch.float32),
    torch.tensor([t.x, t.y, t.z], dtype=torch.float32),
  )


def read_static_tf_graph(bag_path: Path) -> dict[tuple[str, str], torch.Tensor]:
  """Build {(parent, child): UDQ} from /tf_static and first-seen edges of /tf, with KNOWN_STATIC_TF filling gaps."""
  graph: dict[tuple[str, str], torch.Tensor] = {}
  with AnyReader([bag_path], default_typestore=get_typestore(Stores.ROS2_KILTED)) as reader:
    conns = [c for c in reader.connections if c.topic in ('/tf_static', '/tf')]
    for connection, _, raw in reader.messages(connections=conns):
      msg = reader.deserialize(raw, connection.msgtype)
      for tf in msg.transforms:
        key = (_norm_frame(tf.header.frame_id), _norm_frame(tf.child_frame_id))
        if key not in graph:
          graph[key] = _tf_msg_to_udq(tf.transform)
  for k, v in KNOWN_STATIC_TF.items():
    graph.setdefault(k, v)
  return graph


def compose_static_chain(graph: dict[tuple[str, str], torch.Tensor], src: str, dst: str) -> Optional[torch.Tensor]:
  """BFS src→dst over a directed graph; traverses backward edges via udq_conj. Returns UDQ T_src_dst, or None."""
  src, dst = _norm_frame(src), _norm_frame(dst)
  if src == dst:
    return _IDENTITY_UDQ.clone()

  adjacency: dict[str, list[tuple[str, torch.Tensor]]] = defaultdict(list)
  for (p, c), tf in graph.items():
    adjacency[p].append((c, tf))
    adjacency[c].append((p, udq_conj(tf)))

  visited = {src}
  queue: deque[tuple[str, torch.Tensor]] = deque([(src, _IDENTITY_UDQ.clone())])
  while queue:
    node, acc = queue.popleft()
    for nxt, edge in adjacency[node]:
      if nxt in visited:
        continue
      new_acc = udq_mul(acc, edge)
      if nxt == dst:
        return new_acc
      visited.add(nxt)
      queue.append((nxt, new_acc))
  return None


@disk_cache('bag_path', 'lidar_topic', 'odometry_topic')
def extract_sync(bag_path: Path, lidar_topic: str, odometry_topic: str, pose_topic: Optional[str] = None):
  """
  Extract LIDAR frames, reference poses, and odometry data from a ROS2 bag. All messages are synchronized around the
  LIDAR framerate. If no reference poses are available the trajectory is automatically derived using SLAM.
  Odometry is composed with the static TF chain (odom child_frame_id → lidar frame_id) so that the returned
  odometry trajectory expresses the lidar-frame pose in the odom frame, matching the lidar reference trajectory.

  :param bag_path: path to the folder containing the db3 file
  :param lidar_topic: topic the LIDAR messages are published on
  :param odometry_topic: topic the odometry data is published on
  :param pose_topic: topic the optional reference poses are published on
  :return: LIDAR frames as (T, N, 4) and poses / odometry as (T, 8) shaped tensors
  """
  if pose_topic is not None:
    frames, poses, odometry = read_bag_sync(bag_path, (lidar_topic, pose_topic, odometry_topic))
    poses = torch.stack([parse_pose(p) for p in poses])
  else:
    frames, odometry = read_bag_sync(bag_path, (lidar_topic, odometry_topic))
    poses = None

  lidar_frame = _norm_frame(frames[0].header.frame_id)
  odom_child = _norm_frame(odometry[0].child_frame_id)

  odometry = torch.stack([parse_pose(o) for o in odometry])

  graph = read_static_tf_graph(bag_path)
  base_to_lidar = compose_static_chain(graph, odom_child, lidar_frame)
  if base_to_lidar is None:
    available = sorted({f for edge in graph for f in edge})
    raise ValueError(
      f'No static TF chain {odom_child!r} -> {lidar_frame!r} in /tf_static + /tf + KNOWN_STATIC_TF. '
      f'Available frames: {available}'
    )
  odometry = udq_mul(odometry, base_to_lidar)

  frames = [mask_robot(parse_lidar(f)) for f in frames]
  clip = min(f.shape[0] for f in frames)
  frames = torch.stack([f[:clip] for f in frames])

  if poses is None:
    graph = run_slam(frames, voxel_size=0.1, odometry=odometry)
    poses = torch.stack([tf_to_udq(node) for node in graph.nodes])

  return frames, poses, odometry

@disk_cache('bag_paths', 'lidar_topic')
def make_world(bag_paths: list[Path], lidar_topic: str = '/livox/lidar'):
  voxel_size = 0.1
  world = None

  if len(bag_paths) > 1:
    print("WARNING: you are running SDF training on multiple bags, and relying on this code's ability to correctly assemble and orient them into a map.")
    print("It is highly advised you only run SDF training on a single bag, and if you need to fuse multiple bags into one map, to at least inspect it with --sdf-show-map!")
    
  for bag_path in bag_paths:
    bag_frames = next(read_bag_sync(bag_path, (lidar_topic,)))
    bag_frames = [mask_robot(parse_lidar(f)) for f in bag_frames]
    clip = min(f.shape[0] for f in bag_frames)
    bag_frames = torch.stack([f[:clip] for f in bag_frames])

    graph = run_slam(bag_frames, voxel_size=voxel_size)
    bag_poses = torch.stack([tf_to_udq(node) for node in graph.nodes])

    if world is not None:
      tf = register_clouds(bag_frames[0], world, voxel_size=voxel_size)
      alignment_udq = udq_from_qt(
        torch.from_numpy(np.roll(Rotation.from_matrix(tf[:3, :3]).as_quat(), 1).copy()).to(torch.float32),
        torch.from_numpy(tf[:3, 3].copy()).to(torch.float32)
      )
      bag_poses = udq_mul(alignment_udq, bag_poses)

    bag_world = udq_transform_points(bag_poses, bag_frames).flatten(end_dim=-2)
    world = bag_world if world is None else torch.cat([world, bag_world])

  return world


class LocalisationDataset(Dataset):
  """
  Localization data derived from a ROS2 bag file

  Attributes
    ----------
    scans: torch.Tensor (T, N, 4) T point clouds containing N points in the (x, y, z, intensity) format
    poses: torch.Tensor (T, 8) T reference poses in unit dual quaternion representation
    odometry: torch.Tensor (T, 8) T odometry poses in unit dual quaternion representation
    world: torch.Tensor (M, 4) world map as a single set of M points in the (x, y, z, intensity) format
  """

  def __init__(self,
               path: Path,
               lidar_topic: str = '/livox/lidar',
               odometry_topic: str = '/odometry/filtered',
               pose_topic: Optional[str] = None,
               world_pcd: Optional[Path] = None):
    """
    :param path: path to the ROS2 bag folder containing the db3 file
    :param lidar_topic: topic the LIDAR messages are published on
    :param odometry_topic: topic the odometry data is published on
    :param pose_topic: topic the optional reference poses are published on
    :param world_pcd: optional path to a pcd file describing the world map
    """
    self.path: typing.Final = path
    self.lidar_topic: typing.Final = lidar_topic
    self.odometry_topic: typing.Final = odometry_topic
    self.pose_topic: typing.Final = pose_topic
    self.world_pcd: typing.Final = world_pcd

    self.scans, self.poses, self.odometry = extract_sync(path, lidar_topic, odometry_topic, pose_topic)

    if world_pcd is not None:
      self.world = torch.from_numpy(PointCloud.from_path(world_pcd).numpy())
    else:
      self.world = udq_transform_points(self.poses, self.scans).flatten(end_dim=-2)

  def __len__(self):
    return self.scans.shape[0]

  def __getitem__(self, idx):
    return self.world, self.scans[idx], self.poses[idx], self.odometry[idx]

  def __repr__(self):
    return f'LocalisationDataset({self.path}, {self.lidar_topic}, {self.odometry_topic}, {self.pose_topic}, {self.world_pcd})'
