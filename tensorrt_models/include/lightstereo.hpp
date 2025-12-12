#pragma once
#include "trt_model.hpp"

class Lightstereo:public Trt_Model{
public:
    Lightstereo(const std::string &onnx_path, const std::string &engine_path, int stream_number):Trt_Model(onnx_path,engine_path,stream_number){};
    ~Lightstereo(){};
    int doInference(void *input,void *output) override;
private:
    
};
