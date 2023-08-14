all: vector serial cv launch ballistics 

vector:
	colcon build --packages-select cool_vector_type

serial:
	colcon build --packages-select ros2_serial_example --cmake-args -DROS2_SERIAL_PACKAGES="cool_vector_type"


cv:
	colcon build --packages-select cv_tensorrt

launch:
	colcon build --packages-select cv --symlink-install

ballistics:
	colcon build --packages-select tr_ballistics
