#!/bin/bash
cd /home/nvidia/Desktop/TRCV_2023
source install/setup.bash
ros2 run cv_tensorrt cv_tensorrt -d  armor.engine .
