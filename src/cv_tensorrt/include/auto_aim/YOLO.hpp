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

    int OFFSET_INT_YAW = 1800;
    int OFFSET_INT_PITCH = 1800;
    int OFFSET_YAW;
    int OFFSET_PITCH;

    bool debug_;
    rclcpp::Node::SharedPtr nh_;

    cv::Point3f getPose();
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
    void declearAndLoadParameter();


    // solvePnP
    std::vector<cv::Point3f> small_real_armor_points;
    std::vector<cv::Point3f> big_real_armor_points;



    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 2149.398738741313, 0.000000000000, 630.8116709129316,
            0.0000000000000000, 2173.036926782579, 508.2915674836098,
            0.0000000000000000, 0.0000000000000000, 1.0000000000000000);
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5)
            << 0.07598515058037726,
            -5.203545554688566, 0.002682535363310819, -0.001778044841669727, 57.25786468650683);






//    bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir);
//    void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer);
//    void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize);
//    void serialize_engine(unsigned int max_batchsize, bool& is_p6, float& gd, float& gw, std::string& wts_name, std::string& engine_name);
//    void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context);

};

#endif //CV_TENSORRT_YOLO_HPP
