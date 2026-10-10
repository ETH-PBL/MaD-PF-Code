#!/usr/bin/env python3
"""
Online ROS2 localization node using the Differentiable Particle Filter.

Subscribes to a PointCloud2 and an Odometry topic, runs one DPF predict→update→resample
step per synchronized pair, and publishes a PoseWithCovarianceStamped estimate.

Prerequisites:
    1. Build the map from a bag of your environment:
           ros2 run diff_slam build_map --bag data/your-env/ --output map.pt

    2. Train (or reuse) the models:
           models/dpf/model.pt   (preferred — includes fine-tuned observation model)
           models/obs/model.pt   (fallback)

Launch:
    ros2 launch diff_slam localizer.launch.py \
        map_path:=/abs/path/to/map.pt \
        model_path:=/abs/path/to/models/dpf/model.pt

Re-initialize the particle filter at any time by publishing to /initialpose
(geometry_msgs/PoseWithCovarianceStamped), just like AMCL.
"""

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import message_filters
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, TransformStamped
import tf2_ros
from pathlib import Path

from .module import ModuleConfig
from .util import get_tf, apply_tf, tf_log, tf_exp


# ---------------------------------------------------------------------------
# Message parsing  (mirrors bag_parser.py, but for live rclpy messages)
# ---------------------------------------------------------------------------

def parse_lidar_ros2(msg: PointCloud2) -> torch.Tensor:
    """Convert a live PointCloud2 message to an (N, 4) float32 tensor [X, Y, Z, intensity].

    Uses the field descriptor in the message so it works with any LIDAR driver
    regardless of the exact byte layout (Livox, Velodyne, Ouster, …).
    Falls back to zeros for missing fields (e.g. sensors without intensity).
    """
    fields = {f.name: (f.offset, f.datatype) for f in msg.fields}
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)

    def extract(name: str, default: float = 0.0) -> np.ndarray:
        if name not in fields:
            return np.full(len(raw), default, dtype=np.float32)
        offset, dtype_id = fields[name]
        # ROS2 field datatypes: 7 = FLOAT32, 8 = FLOAT64
        np_dtype = np.float64 if dtype_id == 8 else np.float32
        col = np.ascontiguousarray(raw[:, offset: offset + np.dtype(np_dtype).itemsize]).view(np_dtype).flatten()
        return col.astype(np.float32)

    pts = np.stack([extract('x'), extract('y'), extract('z'), extract('intensity')], axis=1)
    # Remove NaN / Inf points that some drivers emit for invalid returns
    valid = np.isfinite(pts).all(axis=1)
    return torch.tensor(pts[valid], dtype=torch.float32)


def parse_pose_ros2(msg) -> torch.Tensor:
    """Convert a ROS2 Odometry or PoseStamped message to a (4, 4) SE(2) matrix."""
    # Unwrap nested .pose fields (Odometry → .pose.pose, PoseStamped → .pose)
    pose = msg
    while hasattr(pose, 'pose'):
        pose = pose.pose

    q = pose.orientation
    P = torch.eye(4, dtype=torch.float32)
    P[0, 0] = 1 - 2 * (q.y ** 2 + q.z ** 2)
    P[0, 1] = 2 * (q.x * q.y - q.z * q.w)
    P[0, 2] = 2 * (q.x * q.z + q.y * q.w)
    P[1, 0] = 2 * (q.x * q.y + q.z * q.w)
    P[1, 1] = 1 - 2 * (q.x ** 2 + q.z ** 2)
    P[1, 2] = 2 * (q.y * q.z - q.x * q.w)
    P[2, 0] = 2 * (q.x * q.z - q.y * q.w)
    P[2, 1] = 2 * (q.y * q.z + q.x * q.w)
    P[2, 2] = 1 - 2 * (q.x ** 2 + q.y ** 2)
    P[0, 3] = pose.position.x
    P[1, 3] = pose.position.y
    P[2, 3] = pose.position.z
    return P


def mask_robot(frame: torch.Tensor, size: float = 0.5) -> torch.Tensor:
    """Drop points that are within the robot body footprint (±size metres)."""
    mask = (frame[:, 0].abs() > size) | (frame[:, 1].abs() > size)
    return frame[mask]


def sample_frame(frame: torch.Tensor, n: int) -> torch.Tensor:
    idx = torch.randperm(frame.shape[0])[:n]
    return frame[idx]


# ---------------------------------------------------------------------------
# DPF localizer node
# ---------------------------------------------------------------------------

class DPFLocalizer(Node):
    """One-step-at-a-time DPF localization, driven by incoming ROS2 messages."""

    # Match the hyperparameters used during DPF training (dpf.py / train_dpf)
    PARTICLE_COUNT = 350
    DOWNSAMPLE = 8000
    PREDICT_NOISE_A = 0.05    # angular std (rad) added to each particle per step
    PREDICT_NOISE_T = 0.10    # translation std (m) added to each particle per step
    WEIGHT_TEMP = 0.0025      # softmax temperature for particle weights
    RESAMPLE_ALPHA = 0.8      # 0 → pure diversity, 1 → pure MAP collapse

    def __init__(self):
        super().__init__('dpf_localizer')

        # --- Parameters -------------------------------------------------------
        self.declare_parameter('map_path',          'map.pt')
        self.declare_parameter('model_path',        'models/dpf/model.pt')
        self.declare_parameter('pointcloud_topic',  '/livox/lidar')
        self.declare_parameter('odometry_topic',    '/odom')
        self.declare_parameter('output_topic',      '/dpf/pose')
        self.declare_parameter('map_frame',         'map')
        self.declare_parameter('base_frame',        'base_link')
        self.declare_parameter('sync_slop',         0.1)   # seconds
        self.declare_parameter('init_spread_angle', 0.10)  # rad std for initial scatter
        self.declare_parameter('init_spread_xy',    0.50)  # m std for initial scatter

        map_path   = self.get_parameter('map_path').value
        model_path = self.get_parameter('model_path').value

        # --- Device -----------------------------------------------------------
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        # --- Map --------------------------------------------------------------
        self.get_logger().info(f'Loading map: {map_path}')
        self.world = torch.load(map_path, map_location=self.device)
        self.get_logger().info(f'Map: {self.world.shape[0]:,} points')

        # --- Observation model ------------------------------------------------
        self.get_logger().info(f'Loading model: {model_path}')
        # Load a config bundle: an observation-model checkpoint, or a full DPF
        # checkpoint (take its observer submodule). Encoder size/params come
        # from the saved config, not the encoder_size parameter.
        _, loaded = ModuleConfig.load_module(Path(model_path))
        obs = loaded.observer if hasattr(loaded, 'observer') else loaded
        self.obs = obs.to(self.device).eval()
        self.get_logger().info('Model ready')

        # --- Particle state ---------------------------------------------------
        self.particles: torch.Tensor | None = None   # (P, 4, 4)
        self.prev_odom: torch.Tensor | None = None   # (4, 4)

        # --- Publisher & TF ---------------------------------------------------
        self.pose_pub = self.create_publisher(
            PoseStamped,
            self.get_parameter('output_topic').value,
            10,
        )
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # --- /initialpose subscriber (like AMCL) ------------------------------
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self._initialpose_cb,
            10,
        )

        # --- Synchronized scan + odometry ------------------------------------
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        pc_sub   = message_filters.Subscriber(self, PointCloud2, self.get_parameter('pointcloud_topic').value, qos_profile=sensor_qos)
        odom_sub = message_filters.Subscriber(self, Odometry,    self.get_parameter('odometry_topic').value,    qos_profile=sensor_qos)

        slop = self.get_parameter('sync_slop').value
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [pc_sub, odom_sub], queue_size=1, slop=slop,
        )
        self.sync.registerCallback(self._sensor_cb)

        self.get_logger().info(
            'DPF localizer ready.\n'
            f'  Listening: {self.get_parameter("pointcloud_topic").value}\n'
            f'             {self.get_parameter("odometry_topic").value}\n'
            f'  Publishing: {self.get_parameter("output_topic").value}\n'
            f'  Publish /initialpose to (re-)initialize particle cloud.'
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_particles(self, x0: torch.Tensor) -> torch.Tensor:
        """Scatter PARTICLE_COUNT particles around pose x0."""
        P = self.PARTICLE_COUNT
        spread_a  = self.get_parameter('init_spread_angle').value
        spread_xy = self.get_parameter('init_spread_xy').value
        noise = get_tf(
            torch.randn(P) * spread_a,
            torch.randn(P, 2) * spread_xy,
        ).to(self.device)
        return x0.unsqueeze(0).expand(P, -1, -1).clone() @ noise

    def _initialpose_cb(self, msg: PoseWithCovarianceStamped):
        """Re-initialize the filter from an /initialpose message."""
        x0 = parse_pose_ros2(msg).to(self.device)
        self.particles = self._init_particles(x0)
        self.prev_odom = None   # reset; will be seeded on next sensor callback
        self.get_logger().info(
            f'Re-initialized at x={x0[0,3]:.2f} y={x0[1,3]:.2f}'
        )

    # ------------------------------------------------------------------
    # One DPF step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _step(self, scan: torch.Tensor, curr_odom: torch.Tensor) -> torch.Tensor:

        """Predict → update → resample; return MAP pose (4, 4)."""
        P = self.PARTICLE_COUNT

        # Relative motion from odometry
        u = torch.linalg.inv(self.prev_odom) @ curr_odom          # (4, 4)
        u_exp = u.unsqueeze(0).expand(P, -1, -1)                   # (P, 4, 4)

        # --- Prediction: apply motion + process noise ---------------------
        noise = get_tf(
            torch.randn(P, device=self.device) * self.PREDICT_NOISE_A,
            torch.randn(P, 2, device=self.device) * self.PREDICT_NOISE_T,
        )
        self.particles = self.particles @ u_exp @ noise             # (P, 4, 4)

        # --- Update: score each particle against the map ------------------
        # Transform the current scan into the world frame under each particle hypothesis
        scan_p = apply_tf(scan, self.particles)                     # (P, M, 4)
        w = self.obs(self.world.unsqueeze(0), scan_p)               # (P, 1)
        w = torch.softmax(w / self.WEIGHT_TEMP, dim=0)             # (P, 1)

        # --- MAP estimate -------------------------------------------------
        p_log = tf_log(self.particles)                              # (P, 4)
        x_log = torch.sum(p_log * w, dim=0)                        # (4,)
        x_map = tf_exp(x_log)                                       # (4, 4)

        # --- Mean-based resample ------------------------------------------
        # Pulls all particles toward the MAP estimate; keeps diversity via alpha
        alpha = self.RESAMPLE_ALPHA
        self.particles = tf_exp(p_log * (1 - alpha) + x_log * alpha)

        return x_map

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------

    def _sensor_cb(self, pc_msg: PointCloud2, odom_msg: Odometry):

        """Synchronized pointcloud + odometry callback."""
        scan = parse_lidar_ros2(pc_msg)
        if scan.shape[0] == 0:
            self.get_logger().warn('Empty pointcloud received, skipping')
            return

        scan = mask_robot(scan)
        if scan.shape[0] > self.DOWNSAMPLE:
            scan = sample_frame(scan, self.DOWNSAMPLE)
        scan = scan.to(self.device)

        curr_odom = parse_pose_ros2(odom_msg).to(self.device)

        # First message: initialize particles around odometry position
        if self.particles is None:
            self.particles = self._init_particles(curr_odom)
            self.prev_odom = curr_odom
            self.get_logger().info('Particle filter initialized from first odometry message.')
            return

        if self.prev_odom is None:
            self.prev_odom = curr_odom
            return

        try:
            x_map = self._step(scan, curr_odom)
        except Exception as exc:
            self.get_logger().error(f'DPF step failed: {exc}')
            return

        self.prev_odom = curr_odom
        self._publish(x_map, pc_msg.header.stamp)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish(self, x_map: torch.Tensor, stamp):
        """Publish PoseWithCovarianceStamped and map→base_link TF."""
        x     = x_map[0, 3].item()
        y     = x_map[1, 3].item()
        cos_t = x_map[0, 0].item()
        sin_t = x_map[1, 0].item()
        theta = float(np.arctan2(sin_t, cos_t))
        qz    = float(np.sin(theta / 2))
        qw    = float(np.cos(theta / 2))

        map_frame  = self.get_parameter('map_frame').value
        base_frame = self.get_parameter('base_frame').value

        # PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp    = stamp
        pose_msg.header.frame_id = map_frame
        pose_msg.pose.position.x    = x
        pose_msg.pose.position.y    = y
        pose_msg.pose.position.z    = 0.0
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        self.pose_pub.publish(pose_msg)

        # TF: map → base_link
        tf_msg = TransformStamped()
        tf_msg.header.stamp    = stamp
        tf_msg.header.frame_id = map_frame
        tf_msg.child_frame_id  = base_frame
        tf_msg.transform.translation.x = x
        tf_msg.transform.translation.y = y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.z    = qz
        tf_msg.transform.rotation.w    = qw
        self.tf_broadcaster.sendTransform(tf_msg)


# ---------------------------------------------------------------------------

def main():
    rclpy.init()
    node = DPFLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
