#pragma once
#include <windows.h>
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <io.h>
#include <onnx/onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <time.h>

using namespace cv;
using namespace Ort;

class OcrClass
{
public:
    OcrClass(const wchar_t* model_path_1, const wchar_t* model_path_2, const wchar_t* model_path_3, std::string fault_path, Env* env = new Env(ORT_LOGGING_LEVEL_WARNING, "test"))
        :session_1(*env, model_path_1, Ort::SessionOptions{ nullptr }), session_2(*env, model_path_2, Ort::SessionOptions{ nullptr }), session_3(*env, model_path_3, Ort::SessionOptions{ nullptr }) {
        Ort::TypeInfo type_info = session_1.GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_node_dims = tensor_info.GetShape();

        Ort::TypeInfo type_info_dec = session_3.GetInputTypeInfo(0);
        auto tensor_info_dec = type_info_dec.GetTensorTypeAndShapeInfo();
        input_node_dims_dec = tensor_info_dec.GetShape();

        faultsPath = fault_path;
    };

    std::string inference(const char* image_path);
    std::string inference(void* pbuffer);

    Mat getMaskCoordinate(Mat& mask, Mat& img);
    Mat getMaskCoordinate2(Mat& mask, Mat& img);
    std::vector<Mat> getMaskCoordinate3(Mat& mask, Mat& img);

    ~OcrClass() {
        session_1.release();
        session_2.release();
        session_3.release();
        std::vector<const char*>().swap(output_node_name_1);
        std::vector<const char*>().swap(input_node_name_1);
        std::vector<const char*>().swap(output_node_name_2);
        std::vector<const char*>().swap(input_node_name_2);
        std::vector<const char*>().swap(output_node_name_dec);
        std::vector<const char*>().swap(input_node_name_dec);
        std::vector<int64_t>().swap(input_node_dims);
        std::vector<int64_t>().swap(input_node_dims_dec);
        memory_info.release();
        std::map<int, std::string>().swap(label_map);
    };

private:
    Session session_1, session_2, session_3;
    int image_size_db = 320;
    int image_size_dec = 64;
    size_t input_tensor_size = 320 * 320 * 3;
    size_t input_tensor_size_dec = 64 * 64 * 3;
    std::vector<const char*> output_node_name_1 = { "dbnet/proba3_sigmoid:0" };
    std::vector<const char*> input_node_name_1 = { "image_input:0" };

    std::vector<const char*> output_node_name_2 = { "step2/dbnet/proba3_sigmoid:0" };
    std::vector<const char*> input_node_name_2 = { "step2/step2_image_input:0" };

    std::vector<const char*> output_node_name_dec = { "decision_out:0" };
    std::vector<const char*> input_node_name_dec = { "cut_image_input:0" };

    std::vector<int64_t> input_node_dims;
    std::vector<int64_t> input_node_dims_dec;
    std::string faultsPath;
    MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    //MemoryInfo memory_info_2 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::map<int, std::string> label_map = {
        {0,"0"},
        {1,"1"},
        {2,"2"},
        {3,"3"},
        {4,"4"},
        {5,"5"},
        {6,"6"},
        {7,"7"},
        {8,"8"},
        {9,"9"},
        {10,"A"},
        {11,"B"},
        {12,"C"},
        {13,"D"},
        {14,"E"},
        {15,"F"},
        {16,"G"},
        {17,"H"},
        {18,"J"},
        {19,"K"},
        {20,"P"},
        {21,"S"},
        {22,"R"},
        {23,"Q"},
    };
};


