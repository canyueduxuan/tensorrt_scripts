cmake_minimum_required(VERSION 3.0.2)
project(infer_sample)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../tensorrt_models tensorrt_models)

find_package(OpenCV REQUIRED)

find_package(yaml-cpp REQUIRED)

add_executable(infer_sample infer_sample.cpp)

target_link_libraries(infer_sample
${OpenCV_LIBS}
yaml-cpp
tensorrt_models
cudart
nvinfer)
