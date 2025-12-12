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

class Trt_Model
{
    public:
        Trt_Model(const std::string &onnx_path, const std::string &engine_path, int stream_number);
        virtual ~Trt_Model();
        virtual int doInference(void *input,void *output) = 0;
    protected:
        int buildEngine();
        int deserializeEngine();
        int getBindingInfo();
        int stream_number_;
        std::string onnx_path_;
        std::string engine_path_;
        nvinfer1::IBuilder *builder;
        nvinfer1::INetworkDefinition* network;
        nvinfer1::IBuilderConfig* config;
        nvinfer1::ICudaEngine* engine;
        std::shared_ptr<nvinfer1::ICudaEngine> enginePtr;
        std::vector<nvinfer1::IExecutionContext*> contexts_;
        std::vector<cudaStream_t> streams_;
        std::vector<std::shared_ptr<samplesCommon::BufferManager>> buffer_managers_;
        
        //bindingInfo
        int numBindings = 0;
        std::vector<std::string> bindingNames;
        std::vector<bool> bindingIsInput;
        std::vector<nvinfer1::Dims> bindingDims;
};