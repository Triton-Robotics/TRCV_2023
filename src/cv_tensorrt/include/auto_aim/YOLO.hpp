//
// Created by hongbin on 3/6/23.
//

#ifndef CV_TENSORRT_YOLO_HPP
#define CV_TENSORRT_YOLO_HPP

#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"

#include <iostream>
#include <chrono>
#include <cmath>


//#include <cstdlib>
//#include <functional>
//#include <memory>
//#include <string>
//#include <stdlib.h>


#include <opencv2/opencv.hpp>
#include "std_msgs/msg/int32.hpp"
#include "cool_vector_type/msg/vector3.hpp"
#include "rclcpp/rclcpp.hpp"
#include "imutils.h"



using namespace nvinfer1;
using std::placeholders::_1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

class YOLO {
public:
    YOLO();
    YOLO(rclcpp::Node::SharedPtr nh)
    {
        nh_ = nh;
        publisher_ = nh_->create_publisher<cool_vector_type::msg::Vector3>("gimbal_data", 10);
    }

    // int OFFSET_INT_YAW = 1800;
    // int OFFSET_INT_PITCH = 1800;
    // int OFFSET_YAW;
    // int OFFSET_PITCH;

    // bool debug_;
    bool isblue;
    rclcpp::Node::SharedPtr nh_;

    cv::Point3f getPose(ARMOR_SIZE size);
    void publishData(double x, double y, double z);

    std::vector<cv::Point2f> final_armor_2Dpoints;
    void load_armor_data();





    ~YOLO();

    bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir);
    void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer);
    void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize);
    void serialize_engine(unsigned int max_batchsize, bool& is_p6, float& gd, float& gw, std::string& wts_name, std::string& engine_name);
    void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context);

private:
    rclcpp::Publisher<cool_vector_type::msg::Vector3>::SharedPtr publisher_;
    void declareAndLoadParameter();


    // solvePnP
    std::vector<cv::Point3f> small_real_armor_points;
    std::vector<cv::Point3f> big_real_armor_points;

    /*
    Camera matrix :

    [[2062.15199    0.       555.84843]
    [   0.      2051.33571  451.03496]
    [   0.         0.         1.     ]]
    */

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
            2062.15199, 0.0,    555.84843,
            0.0,        2051.33571, 451.03496,
            0.0,        0.0,        1.0
    );

    /*
    dist :

    [[-0.20666  0.48607 -0.01596 -0.02134  3.17586]]
    */
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 
            -0.20666, 0.48607, -0.01596, -0.02134, 3.17586
    );






//    bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir);
//    void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer);
//    void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize);
//    void serialize_engine(unsigned int max_batchsize, bool& is_p6, float& gd, float& gw, std::string& wts_name, std::string& engine_name);
//    void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context);

};

#endif //CV_TENSORRT_YOLO_HPP
