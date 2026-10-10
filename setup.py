from setuptools import find_packages, setup

package_name = 'diff_slam'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/localizer.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='diff-slam maintainer',
    maintainer_email='user@example.com',
    description='Differentiable Particle Filter 3D LIDAR localization for ROS2 Humble.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'localizer = diff_slam.ros2_localizer:main',
            'build_map = diff_slam.build_map:main',
        ],
    },
)
