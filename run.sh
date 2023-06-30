#colcon build
#colcon build --symlink-install --packages-select cv_tensorrt
source install/setup.bash
# ros2 run cv_tensorrt cv_tensorrt -d  armor.engine .

ros2 run cv_tensorrt cv_tensorrt
