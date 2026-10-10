from typing import Iterable, Optional
from collections import deque

import numpy as np
import torch.utils.dlpack
import open3d as o3d
import open3d.core as o3c
from open3d.geometry import PointCloud, KDTreeSearchParamHybrid
from open3d.utility import Vector3dVector
from open3d.pipelines.registration import (
    PoseGraph, PoseGraphNode, registration_generalized_icp, PoseGraphEdge,
    GlobalOptimizationOption, GlobalOptimizationLevenbergMarquardt,
    GlobalOptimizationConvergenceCriteria, global_optimization
)
from tqdm import tqdm
import torch


def run_slam(
    frames: Iterable[torch.Tensor],
    voxel_size: float,
    odometry: Optional[torch.Tensor] = None,
    close_loop: bool = False,
    map_window: Optional[int] = None,
) -> PoseGraph:
    """
    Runs ICP SLAM with scan-to-map registration using Open3D Tensor API and GPU.

    :param frames: Iterable of point cloud frames (N, 3+)
    :param voxel_size: Voxel size for downsampling and ICP thresholds
    :param odometry: Optional (T, 4, 4) or (T, 8) udq prior poses
    :param close_loop: Whether to perform global loop closure
    :param map_window: Number of previous frames to use for local map registration. If None, use all previous frames.
    :return: Optimized PoseGraph
    """
    device = o3c.Device("CPU:0")

    points = []
    for frame in tqdm(frames, desc='preprocessing scans'):
        # Convert to Open3D Tensor PointCloud
        if isinstance(frame, torch.Tensor):
            try:
                # Use DLPack if possible for zero-copy
                o3d_points = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(frame[..., :3].contiguous()))
            except Exception:
                o3d_points = o3c.Tensor(frame[..., :3].numpy(), device=device)
        else:
            o3d_points = o3c.Tensor(frame[..., :3], device=device)
        
        o3d_points = o3d_points.to(device)
        
        pcd = o3d.t.geometry.PointCloud(o3d_points)
        pcd = pcd.voxel_down_sample(voxel_size)
        pcd.estimate_normals(max_nn=30, radius=voxel_size * 4)
        points.append(pcd)

    # Convert odometry to matrices if provided
    odom_mats = None
    if odometry is not None:
        if odometry.shape[-1] == 8:
            from .pose import udq_to_matrix
            odom_mats = udq_to_matrix(odometry).numpy()
        else:
            odom_mats = odometry.numpy()

    graph = PoseGraph()
    current_pose = np.eye(4)
    graph.nodes.append(PoseGraphNode(current_pose))

    # Maintain transformed points for local map
    transformed_window = deque(maxlen=map_window if map_window is not None else None)
    
    # First frame
    pcd0_world = points[0].clone()
    pcd0_world.transform(o3c.Tensor(graph.nodes[0].pose, device=device))
    transformed_window.append(pcd0_world)
    
    target_map = pcd0_world

    for i in tqdm(range(len(points) - 1), desc='registering scans'):
        init_guess = graph.nodes[i].pose
        if odom_mats is not None:
            rel_odom = np.linalg.inv(odom_mats[i]) @ odom_mats[i+1]
            init_guess = graph.nodes[i].pose @ rel_odom

        if map_window is not None:
            target_map = transformed_window[0].clone()
            for j in range(1, len(transformed_window)):
                target_map += transformed_window[j]
            target_map = target_map.voxel_down_sample(voxel_size)
            target_map.estimate_normals(max_nn=30, radius=voxel_size * 4)

        # Run ICP
        reg_result = o3d.t.pipelines.registration.icp(
            source=points[i + 1],
            target=target_map,
            max_correspondence_distance=voxel_size * 2,
            init_source_to_target=o3c.Tensor(init_guess, device=device),
            estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
        )

        new_pose_world = reg_result.transformation.cpu().numpy()
        graph.nodes.append(PoseGraphNode(new_pose_world))
        
        rel_tf = np.linalg.inv(graph.nodes[i].pose) @ graph.nodes[i+1].pose
        
        # Information matrix (Tensor API)
        # Optimization: use a slightly more downsampled source for information matrix if it's too large
        info_source = points[i + 1]
        if info_source.point.positions.shape[0] > 2000:
            info_source = info_source.voxel_down_sample(voxel_size * 2)

        info_tensor = o3d.t.pipelines.registration.get_information_matrix(
            source=info_source,
            target=target_map,
            max_correspondence_distance=voxel_size * 2,
            transformation=reg_result.transformation
        )
        info = info_tensor.cpu().numpy()

        graph.edges.append(PoseGraphEdge(i, i + 1, rel_tf, info, uncertain=False))

        # Update window and map
        pcd_next_world = points[i+1].clone()
        pcd_next_world.transform(o3c.Tensor(new_pose_world, device=device))
        
        if map_window is not None:
            transformed_window.append(pcd_next_world)
        else:
            target_map += pcd_next_world
            # Downsample more frequently to keep map size manageable and ICP fast
            if (i + 1) % 5 == 0:
                target_map = target_map.voxel_down_sample(voxel_size)
                target_map.estimate_normals(max_nn=30, radius=voxel_size * 4)

    if close_loop:
        n = len(points) - 1
        init_loop = np.linalg.inv(graph.nodes[0].pose) @ graph.nodes[n].pose
        
        reg_result = o3d.t.pipelines.registration.icp(
            source=points[n],
            target=points[0],
            max_correspondence_distance=voxel_size * 4,
            init_source_to_target=o3c.Tensor(init_loop, device=device),
            estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        
        if reg_result.fitness > 0.4:
            info_tensor = o3d.t.pipelines.registration.get_information_matrix(
                source=points[n],
                target=points[0],
                max_correspondence_distance=voxel_size * 4,
                transformation=reg_result.transformation
            )
            graph.edges.append(PoseGraphEdge(n, 0, reg_result.transformation.cpu().numpy(), info_tensor.cpu().numpy(), uncertain=True))

        global_optimization(
            graph,
            GlobalOptimizationLevenbergMarquardt(),
            GlobalOptimizationConvergenceCriteria(),
            GlobalOptimizationOption(reference_node=0)
        )

    return graph

def register_clouds(source: torch.Tensor, target: torch.Tensor, voxel_size: float, init: np.ndarray = np.eye(4)) -> np.ndarray:
  source_pcd = PointCloud()
  source_pcd.points = Vector3dVector(source[..., :3].numpy())
  source_pcd.estimate_normals(KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30))

  if target.shape[0] > 100000:
    target = target[torch.randperm(target.shape[0])[:100000]]

  target_pcd = PointCloud()
  target_pcd.points = Vector3dVector(target[..., :3].numpy())
  target_pcd.estimate_normals(KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30))

  return registration_generalized_icp(
    source=source_pcd,
    target=target_pcd,
    max_correspondence_distance=voxel_size * 10,
    init=init,
  ).transformation