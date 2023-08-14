import os

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node
from launch_ros.actions import PushRosNamespace


def generate_launch_description():

    ld = LaunchDescription()

    yolo_node = Node(
        package='cv_tensorrt',
        executable='cv_tensorrt',
        parameters=[{
            "is_blue": True,
        }]

    )

    ballistics_node = Node(
        package='tr_ballistics',
        executable='ballistics'
    )

    serial_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros2_serial_example'),
                'launch/custom.launch.py'))
    )


    ld.add_action(yolo_node)
    ld.add_action(ballistics_node)
    ld.add_action(serial_node)

    return ld
