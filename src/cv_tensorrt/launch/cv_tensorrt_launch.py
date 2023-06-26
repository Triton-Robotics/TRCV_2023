from distutils.command.config import config
from click import launch
from ament_index_python import get_package_share_directory
from launch import LaunchDescription

from launch_ros.actions import Node

import os
def generate_launch_description():

    config = os.path.join (
        get_package_share_directory('cv_tensorrt'),
        'config',
        'autoAim.yaml'

    )
    return LaunchDescription(
        [
            Node(
                package="cv_tensorrt",
                namespace='',
                executable="cv_tensorrt",
                name='cv_tensorrt',
                parameters=[config]
            )
        ]
    )
