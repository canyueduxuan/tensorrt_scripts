#pragma once
#include "trt_model.hpp"

class Crestereo:public Trt_Model{
public:
    Crestereo(const std::string &onnx_path, const std::string &engine_path, int stream_number):Trt_Model(onnx_path,engine_path,stream_number){};
    ~Crestereo(){};
    int doInference(void *input,void *output) override;
private:
    
};
