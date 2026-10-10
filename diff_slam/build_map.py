#!/usr/bin/env python3
"""
Build the localization map, either from a ROS2 bag or an existing PCD file.

The output map.pt is a (N, 4) float32 tensor [X, Y, Z, intensity] in world frame.

Usage — from a ROS2 bag (runs ICP SLAM to align scans):
    ros2 run diff_slam build_map --bag data/your-environment/ --output map.pt
    ros2 run diff_slam build_map --bag data/my-env/ --output map.pt \\
        --lidar-topic /velodyne_points --odometry-topic /odom

Usage — from an existing PCD file (already in world frame, no SLAM needed):
    ros2 run diff_slam build_map --pcd your_map.pcd --output map.pt
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from .bag_parser import extract_sync
from .pose import q_conj, q_mul, udq_transform_points


# PCD type-code → numpy dtype
_PCD_TYPES = {
    ('F', 4): np.float32,
    ('F', 8): np.float64,
    ('I', 1): np.int8,  ('I', 2): np.int16,  ('I', 4): np.int32,
    ('U', 1): np.uint8, ('U', 2): np.uint16, ('U', 4): np.uint32,
}


def read_pcd(path: Path) -> dict[str, np.ndarray]:
    """Parse a PCD file (ASCII or binary) and return every field as a numpy array.

    Supports the standard PCD 0.7 format written by PCL, KISS-ICP, LIO-SAM, etc.
    Multi-count fields (COUNT > 1) are flattened into separate columns named
    'field_0', 'field_1', …
    """
    with open(path, 'rb') as f:
        # --- Header -----------------------------------------------------------
        header = {}
        while True:
            line = f.readline().decode('ascii', errors='replace').strip()
            if line.startswith('#'):
                continue
            key, *vals = line.split()
            header[key.upper()] = vals
            if key.upper() == 'DATA':
                data_encoding = vals[0].lower()
                break

        fields  = header['FIELDS']
        sizes   = [int(s) for s in header['SIZE']]
        types   = header['TYPE']
        counts  = [int(c) for c in header['COUNT']]
        n_pts   = int(header['POINTS'][0])

        # --- Build a structured dtype for one point ---------------------------
        dt_fields = []
        for name, size, typ, count in zip(fields, sizes, types, counts):
            np_dtype = _PCD_TYPES.get((typ, size))
            if np_dtype is None:
                raise ValueError(f'Unsupported PCD field type: {typ}{size*8}')
            if count == 1:
                dt_fields.append((name, np_dtype))
            else:
                for i in range(count):
                    dt_fields.append((f'{name}_{i}', np_dtype))

        point_dtype = np.dtype(dt_fields)

        # --- Read data --------------------------------------------------------
        if data_encoding == 'ascii':
            raw = f.read()
            data = np.fromstring(raw, sep=' ', dtype=np.float64)
            data = data.reshape(n_pts, -1)
            # re-cast each column to its declared dtype
            arrays = {}
            col = 0
            for name, size, typ, count in zip(fields, sizes, types, counts):
                np_dtype = _PCD_TYPES[(typ, size)]
                if count == 1:
                    arrays[name] = data[:, col].astype(np_dtype)
                    col += 1
                else:
                    for i in range(count):
                        arrays[f'{name}_{i}'] = data[:, col].astype(np_dtype)
                        col += 1
            return arrays

        elif data_encoding in ('binary', 'binary_compressed'):
            if data_encoding == 'binary_compressed':
                import lzf  # pip install python-lzf
                compressed_size, uncompressed_size = struct.unpack('<II', f.read(8))
                raw = lzf.decompress(f.read(compressed_size), uncompressed_size)
                data = np.frombuffer(raw, dtype=point_dtype)
            else:
                data = np.frombuffer(f.read(n_pts * point_dtype.itemsize), dtype=point_dtype)

            return {name: data[name] for name in data.dtype.names}

        else:
            raise ValueError(f'Unknown PCD data encoding: {data_encoding}')


def save_g2o(poses: torch.Tensor, path: Path) -> None:
    """Save (N, 8) udq poses as VERTEX_SE3:QUAT g2o file (1-indexed vertex IDs)."""
    with open(path, 'w') as f:
        for i, p in enumerate(poses):
            qr, qd = p[:4], p[4:]
            t = (2 * q_mul(qd, q_conj(qr)))[1:]
            qw, qx, qy, qz = qr
            f.write(f'VERTEX_SE3:QUAT {i + 1} {t[0]:.10f} {t[1]:.10f} {t[2]:.10f} '
                    f'{qx:.10f} {qy:.10f} {qz:.10f} {qw:.10f}\n')


def from_bag(
    bag_path: Path,
    output_path: Path,
    lidar_topic: str = '/livox/lidar',
    odometry_topic: str = '/odometry/filtered',
):
    print(f'Parsing bag: {bag_path}')
    print(f'  lidar topic:    {lidar_topic}')
    print(f'  odometry topic: {odometry_topic}')

    frames, poses, _ = extract_sync(bag_path, lidar_topic, odometry_topic)

    print(f'Extracted {frames.shape[0]} frames, {frames.shape[1]} pts/frame')
    print('Building world map...')

    world = udq_transform_points(poses, frames).flatten(end_dim=-2)

    print(f'Map has {world.shape[0]:,} points')
    torch.save(world, output_path)
    print(f'Saved to {output_path}')

    g2o_path = bag_path / 'pose_graph.g2o'
    save_g2o(poses, g2o_path)
    print(f'Saved poses to {g2o_path}')


def from_pcd(pcd_path: Path, output_path: Path):
    """Convert an existing PCD file (already in world frame) to map.pt.

    Reads every field directly from the PCD header so that scalar fields such as
    'intensity' are preserved exactly, regardless of whether the file is ASCII or
    binary.  Falls back to zeros when no intensity-like field is present (the BEV
    encoder will still work — geometry dominates).

    Common field names tried for intensity (in order):
        intensity, i, reflectivity, reflectance, amplitude
    """
    print(f'Loading PCD: {pcd_path}')
    fields = read_pcd(pcd_path)
    print(f'  Fields found: {list(fields.keys())}')

    # XYZ — required
    for req in ('x', 'y', 'z'):
        if req not in fields:
            raise ValueError(f"PCD is missing required field '{req}'")

    xyz = np.stack([
        fields['x'].astype(np.float32),
        fields['y'].astype(np.float32),
        fields['z'].astype(np.float32),
    ], axis=1)                                                      # (N, 3)

    # Intensity — try common names
    intensity_candidates = ('intensity', 'i', 'reflectivity', 'reflectance', 'amplitude')
    intensity_col = next((f for f in intensity_candidates if f in fields), None)

    if intensity_col:
        intensity = fields[intensity_col].astype(np.float32).reshape(-1, 1)
        print(f'  Using field "{intensity_col}" as intensity')
    else:
        intensity = np.zeros((len(xyz), 1), dtype=np.float32)
        print('  No intensity field found — setting to 0 (geometry-only matching)')

    # Remove NaN / Inf
    world_np = np.concatenate([xyz, intensity], axis=1)
    valid = np.isfinite(world_np).all(axis=1)
    world_np = world_np[valid]

    world = torch.tensor(world_np, dtype=torch.float32)
    print(f'Map has {world.shape[0]:,} points')
    torch.save(world, output_path)
    print(f'Saved to {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Build diff-slam localization map')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bag', type=Path, help='ROS2 bag folder (containing *.db3); runs ICP SLAM')
    source.add_argument('--pcd', type=Path, help='Existing PCD file already in world frame')

    parser.add_argument('--output', type=Path, default=Path('map.pt'), help='Output map file (default: map.pt)')
    parser.add_argument('--lidar-topic',    default='/livox/lidar',        help='(bag only) PointCloud2 topic')
    parser.add_argument('--odometry-topic', default='/odometry/filtered',  help='(bag only) Odometry topic')
    args = parser.parse_args()

    if args.pcd:
        from_pcd(args.pcd, args.output)
    else:
        from_bag(args.bag, args.output, args.lidar_topic, args.odometry_topic)


if __name__ == '__main__':
    main()
