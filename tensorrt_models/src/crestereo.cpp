#include "crestereo.hpp"
Crestereo::Crestereo(const std::string& onnx_path,const std::string& engine_path) {
    initLibNvInferPlugins(&sample::gLogger, "");
    this->onnx_path_ = onnx_path;
    this->engine_path_ = engine_path;
    if(access(engine_path.c_str(), F_OK) == 0)
    {
        std::cout << "engine file find" << std::endl;
        if(deserializeEngine() != 0)
        {
            std::cout << "deserialize engine file failed" << std::endl;
            std::exit(0);
        }
    }
    else
    {
        std::cout << "engine file not find,build engine file from onnx" << std::endl;
        if (access(onnx_path.c_str(), F_OK) == 0) {
            std::cout << "onnx file find" << std::endl;
            if(buildEngine() != 0)
            {
                std::cout << "build engine file from onnx failed" << std::endl;
                std::exit(0);
            }
        } else {
            std::cout << "onnx file not find" << std::endl;
            std::exit(0);
        }
    }
    this->context = this->engine->createExecutionContext();
    if(!this->context) {
        std::cerr << "Failed to create execution context." << std::endl;
        std::exit(0);
    }
    cudaStreamCreate(&this->stream_);
    if(this->stream_ == nullptr)
    {
        std::cout << "cudaStreamCreate failed" << std::endl;
        std::exit(0);
    }
    std::shared_ptr<nvinfer1::ICudaEngine> enginePtr(this->engine);
    this->buffer_manager = std::make_unique<samplesCommon::BufferManager>(enginePtr,0,this->context);
}

Crestereo::~Crestereo() {
    // 1. 销毁执行上下文
    if (this->context) {
        this->context->destroy();  // 调用 TensorRT 的 destroy 方法来销毁上下文
        this->context = nullptr;   // 将指针置空
    }

    // 2. 销毁 CUDA 流
    if (this->stream_) {
        cudaStreamDestroy(this->stream_);  // 使用 cudaStreamDestroy 销毁流
    }

    // 3. 释放 TensorRT 引擎
    if (this->engine) {
        this->engine->destroy();  // 调用 TensorRT 的 destroy 方法来销毁引擎
        this->engine = nullptr;   // 将指针置空
    }

    // 4. 销毁 BufferManager
    if (this->buffer_manager) {
        this->buffer_manager.reset();  // 使用 reset 销毁 BufferManager
    }
}

int Crestereo::buildEngine() {
    this->builder = nvinfer1::createInferBuilder(sample::gLogger);
    if(!this->builder) {
        std::cerr << "Failed to create TensorRT builder." << std::endl;
        return -1;
    }
    this->network = this->builder->createNetworkV2(1U << (int) nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    if(!this->network) {
        std::cerr << "Failed to create TensorRT network." << std::endl;
        return -2;
    }
    auto parser = nvonnxparser::createParser(*this->network, sample::gLogger);
    if(!parser) {
        std::cerr << "Failed to create ONNX parser." << std::endl;
        return -3;
    }
    if(!parser->parseFromFile(this->onnx_path_.c_str(), 2)) {
        std::cerr << "Failed to parse ONNX file." << std::endl;
        return -4;
    }
    this->config = this->builder->createBuilderConfig();
    if(!this->config) {
        std::cerr << "Failed to create TensorRT builder config." << std::endl;
        return -5;
    }
    auto profile = builder->createOptimizationProfile();
    if (profile == nullptr){
        std::cerr << "createOptimizationProfile failed" << std::endl;
        return -6;
    }
    nvinfer1::Dims4 input_dims{1, 3, 120, 160};
    profile->setDimensions("left", nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions("left", nvinfer1::OptProfileSelector::kOPT, input_dims);
    profile->setDimensions("left", nvinfer1::OptProfileSelector::kMAX, input_dims);
    profile->setDimensions("right", nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions("right", nvinfer1::OptProfileSelector::kOPT, input_dims);
    profile->setDimensions("right", nvinfer1::OptProfileSelector::kMAX, input_dims);
    this->config->addOptimizationProfile(profile);
    this->config->setMaxWorkspaceSize(6ULL * 1024 * 1024 * 1024);
    this->config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,6ULL * 1024 * 1024 * 1024);
    this->config->setFlag(nvinfer1::BuilderFlag::kFP16);
    this->config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    this->engine = this->builder->buildEngineWithConfig(*this->network, *this->config);
    if(!this->engine) {
        std::cerr << "Failed to build TensorRT engine." << std::endl;
        return -7;
    }
    std::ofstream engineFile(this->engine_path_, std::ios::binary);
    if(!engineFile) {
        std::cerr << "Failed to open engine file for writing." << std::endl;
        return -8;
    }
    nvinfer1::IHostMemory* engineData = this->engine->serialize();
    engineFile.write(static_cast<const char*>(engineData->data()), engineData->size());
    engineFile.close();
    engineData->destroy();
    return 0;
}

int Crestereo::deserializeEngine() {
    std::ifstream engineFile(this->engine_path_, std::ios::binary);
    if(!engineFile) {
        std::cerr << "Failed to open engine file for reading." << std::endl;
        return -1;
    }
    engineFile.seekg(0, std::ios::end);
    size_t fsize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    engineFile.close();
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(sample::gLogger);
    if(!runtime) {
        std::cerr << "Failed to create TensorRT runtime." << std::endl;
        return -2;
    }
    this->engine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
    if(!this->engine) {
        std::cerr << "Failed to deserialize TensorRT engine." << std::endl;
        return -3;
    }
    runtime->destroy();
    return 0;
}
