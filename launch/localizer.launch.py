from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map_path',           default_value='map.pt',
                              description='Path to the map.pt tensor file'),
        DeclareLaunchArgument('model_path',         default_value='models/dpf/model.pt',
                              description='Path to the DPF model checkpoint'),
        DeclareLaunchArgument('pointcloud_topic',   default_value='/livox/lidar',
                              description='PointCloud2 input topic'),
        DeclareLaunchArgument('odometry_topic',     default_value='/odom',
                              description='Odometry input topic'),
        DeclareLaunchArgument('output_topic',       default_value='/dpf/pose',
                              description='PoseWithCovarianceStamped output topic'),
        DeclareLaunchArgument('map_frame',          default_value='map',
                              description='Fixed map frame id'),
        DeclareLaunchArgument('base_frame',         default_value='base_link',
                              description='Robot base frame id'),
        DeclareLaunchArgument('encoder_size',       default_value='100',
                              description='BEV grid size in cells (100 = 15 m window at 0.15 m/cell)'),
        DeclareLaunchArgument('sync_slop',          default_value='0.1',
                              description='ApproximateTimeSynchronizer slop in seconds'),
        DeclareLaunchArgument('init_spread_xy',     default_value='0.5',
                              description='Initial particle spread in XY (metres, std dev)'),
        DeclareLaunchArgument('init_spread_angle',  default_value='0.1',
                              description='Initial particle spread in heading (radians, std dev)'),

        Node(
            package='diff_slam',
            executable='localizer',
            name='dpf_localizer',
            output='screen',
            parameters=[{
                'map_path':          LaunchConfiguration('map_path'),
                'model_path':        LaunchConfiguration('model_path'),
                'pointcloud_topic':  LaunchConfiguration('pointcloud_topic'),
                'odometry_topic':    LaunchConfiguration('odometry_topic'),
                'output_topic':      LaunchConfiguration('output_topic'),
                'map_frame':         LaunchConfiguration('map_frame'),
                'base_frame':        LaunchConfiguration('base_frame'),
                'encoder_size':      LaunchConfiguration('encoder_size'),
                'sync_slop':         LaunchConfiguration('sync_slop'),
                'init_spread_xy':    LaunchConfiguration('init_spread_xy'),
                'init_spread_angle': LaunchConfiguration('init_spread_angle'),
            }],
        ),
    ])
