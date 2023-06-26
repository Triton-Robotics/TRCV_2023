# adapted the CUDA and tensorrt part from
# https://github.com/wang-xinyu/tensorrtx/blob/master/yolov5/CMakeLists.txt

cmake_minimum_required(VERSION 3.23)
project(cv_tensorrt)

add_definitions(-std=c++14)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

# ROS2
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cool_vector_type REQUIRED)

# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
# TODO(Call for PR): make TRT path configurable from command line
include_directories(/home/nvidia/TensorRT-8.2.5.1/include/)
link_directories(/home/nvidia/TensorRT-8.2.5.1/lib/)

include_directories(lib/yolov5/src/)
include_directories(lib/yolov5/plugin/)
file(GLOB_RECURSE SRCS lib/yolov5/src/*.cpp lib/yolov5/src/*.cu)
file(GLOB_RECURSE PLUGIN_SRCS lib/yolov5/plugin/*.cu)

add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins nvinfer cudart)

# Opencv
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# Huaray SDK
find_library( CAMERA_LIB NAMES MVSDK HINTS camera_lib/arm64 REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp src/Camera.cpp src/YOLO.cpp ${SRCS} )

ament_target_dependencies(${PROJECT_NAME} rclcpp std_msgs cool_vector_type)
#
target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} cudart)
target_link_libraries(${PROJECT_NAME}  myplugins)
target_link_libraries(${PROJECT_NAME} ${CAMERA_LIB} ${OpenCV_LIBS} rt pthread)


target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

install(TARGETS ${PROJECT_NAME}
        DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY
        launch
        DESTINATION share/${PROJECT_NAME}/
        )

install(DIRECTORY
        config
        DESTINATION share/${PROJECT_NAME}/
        )

install(DIRECTORY
        lib/my_plugin
        DESTINATION lib
        )

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
