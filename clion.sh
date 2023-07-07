#!/bin/bash
source install/setup.bash
colcon build --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G "Unix Makefiles"
source install/setup.bash
clion

