"""Convert a single ROS2 bag to KITTI odometry format.

Output layout:  <bag>/eval/
    velodyne/000000.bin ...   float32 (N, 4) x y z intensity
    poses.txt                 3x4 SE3 rows, one per frame
    times.txt                 timestamps in seconds
    calib.txt                 identity Tr (LiDAR-only, no camera)
"""

import argparse
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore, Stores
from tqdm import tqdm

from .bag_parser import extract_sync
from .pose import udq_to_matrix


_IDENTITY_3x4 = '1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 ' \
                '0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 ' \
                '0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00'

_CALIB = (
    f'P0: {_IDENTITY_3x4}\n'
    f'P1: {_IDENTITY_3x4}\n'
    f'P2: {_IDENTITY_3x4}\n'
    f'P3: {_IDENTITY_3x4}\n'
    f'Tr: {_IDENTITY_3x4}\n'
)


def _lidar_timestamps(bag_path: Path, lidar_topic: str) -> list[float]:
    typestore = get_typestore(Stores.ROS2_KILTED)
    ts0 = None
    timestamps = []
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == lidar_topic]
        for _, ts_ns, _ in reader.messages(connections=conns):
            if ts0 is None:
                ts0 = ts_ns
            timestamps.append((ts_ns - ts0) * 1e-9)
    return timestamps


def convert(
    bag_path: Path,
    lidar_topic: str = '/livox/lidar',
    odometry_topic: str = '/odometry/filtered',
    pose_topic: str | None = None,
) -> None:
    out = bag_path / 'eval'
    velodyne_dir = out / 'velodyne'
    velodyne_dir.mkdir(parents=True, exist_ok=True)

    print('extracting frames and poses...')
    frames, poses, _ = extract_sync(bag_path, lidar_topic, odometry_topic, pose_topic)
    # frames: (T, N, 4) float32;  poses: (T, 8) UDQ

    print('extracting timestamps...')
    timestamps = _lidar_timestamps(bag_path, lidar_topic)
    T = frames.shape[0]
    if len(timestamps) != T:
        print(f'  warning: {len(timestamps)} lidar timestamps vs {T} frames — truncating to min')
        n = min(len(timestamps), T)
        timestamps = timestamps[:n]
        frames = frames[:n]
        poses = poses[:n]

    mats = udq_to_matrix(poses)  # (T, 4, 4)

    print(f'writing {T} frames to {out}')
    for i in tqdm(range(T)):
        bin_path = velodyne_dir / f'{i:06d}.bin'
        frames[i].numpy().astype(np.float32).tofile(bin_path)

    poses_path = out / 'poses.txt'
    with open(poses_path, 'w') as f:
        for mat in mats:
            row = mat[:3, :].flatten().tolist()
            f.write(' '.join(f'{v:.6e}' for v in row) + '\n')

    times_path = out / 'times.txt'
    with open(times_path, 'w') as f:
        for t in timestamps:
            f.write(f'{t:.6f}\n')

    calib_path = out / 'calib.txt'
    calib_path.write_text(_CALIB)

    print(f'done → {out}')
    print(f'  {T} scans, poses.txt, times.txt, calib.txt')


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert ROS2 bag to KITTI odometry format')
    parser.add_argument('bag', type=Path, help='path to ROS2 bag folder')
    parser.add_argument('--lidar-topic', default='/livox/lidar')
    parser.add_argument('--odometry-topic', default='/odometry/filtered')
    parser.add_argument('--pose-topic', default=None,
                        help='optional GT pose topic; if omitted, SLAM is used')
    args = parser.parse_args()

    convert(args.bag, args.lidar_topic, args.odometry_topic, args.pose_topic)


if __name__ == '__main__':
    main()
