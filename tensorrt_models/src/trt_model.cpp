#include "trt_model.hpp"
Trt_Model::Trt_Model(const std::string &onnx_path, const std::string &engine_path, int stream_number){
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
    //多流推理
    this->stream_number_ = stream_number;
    this->contexts_.resize(this->stream_number_);
    this->streams_.resize(this->stream_number_);
    this->buffer_managers_.resize(this->stream_number_);
    this->enginePtr = std::shared_ptr<nvinfer1::ICudaEngine>(
        this->engine,
        [](nvinfer1::ICudaEngine *p)
        { if(p) p->destroy(); });
    for (int i = 0; i < this->stream_number_; i++)
    {
        this->contexts_[i] = this->engine->createExecutionContext();
        if (!contexts_[i]) {
            std::cerr << "Failed to create execution context." << i << std::endl;
            std::exit(0);
        }
        cudaStreamCreate(&streams_[i]);
        if(this->streams_[i] == nullptr)
        {
            std::cout << "cudaStreamCreate failed" << i << std::endl;
            std::exit(0);
        }

        this->buffer_managers_[i] = std::make_unique<samplesCommon::BufferManager>(
            this->enginePtr, 0, this->contexts_[i]
        );
    }
    getBindingInfo();
}

int Trt_Model::getBindingInfo()
{
    this->numBindings = this->engine->getNbBindings();
    std::cout << "Number of Bindings: " << this->numBindings << std::endl;

    // 遍历所有绑定，分别获取输入和输出的名称及维度
    for (int i = 0; i < this->numBindings; ++i) {
        // 获取绑定的名称
        this->bindingNames.push_back(this->engine->getBindingName(i));
        // 获取绑定的类型（输入或输出）
        this->bindingDims.push_back(this->engine->getBindingDimensions(i));
        // 判断当前是输入还是输出
        this->bindingIsInput.push_back(this->engine->bindingIsInput(i));
        if (this->bindingIsInput[i])
        {
            std::cout << "Input " << i << ": " << this->bindingNames[i] << std::endl;
        }
        else
        {
            std::cout << "Output " << i << ": " << this->bindingNames[i] << std::endl;
        }
        std::cout << "Dimensions: (";
        for (int j = 0; j < this->bindingDims[i].nbDims; ++j) {
            std::cout << this->bindingDims[i].d[j];
            if (j < this->bindingDims[i].nbDims - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
}

Trt_Model::~Trt_Model() {
    // 1. 先销毁 BufferManager
    for (int i = 0; i < this->stream_number_; i++) {
        if (this->buffer_managers_[i]) {
            this->buffer_managers_[i].reset();
        }
    }

    // 2. 再销毁 ExecutionContext
    for (int i = 0; i < this->stream_number_; i++) {
        if (this->contexts_[i]) {
            this->contexts_[i]->destroy();
            this->contexts_[i] = nullptr;
        }
    }

    // 3. 销毁 CUDA 流
    for (int i = 0; i < this->stream_number_; i++) {
        if (this->streams_[i]) {
            cudaStreamDestroy(this->streams_[i]);
        }
    }

    // 4. 最后释放 enginePtr
    this->enginePtr.reset();
}

int Trt_Model::buildEngine() {
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

    for (int i = 0; i < network->getNbInputs(); ++i) {
        nvinfer1::ITensor* input = network->getInput(i);
        auto dims = input->getDimensions();
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims);
    }

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

int Trt_Model::deserializeEngine() {
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