#pragma once
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include "logger.h"
#include <unistd.h>
#include "buffers.h"

class Crestereo {
public:
    Crestereo(const std::string& onnx_path,const std::string& engine_path);
    ~Crestereo();
    bool infer(const std::vector<float>& inputData, std::vector<float>& outputData);
public:
    int buildEngine();
    int deserializeEngine();
    std::string onnx_path_;
    std::string engine_path_;
    nvinfer1::IBuilder *builder;
    nvinfer1::INetworkDefinition* network;
    nvinfer1::IBuilderConfig* config;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream_;
    std::shared_ptr<samplesCommon::BufferManager> buffer_manager;
};
