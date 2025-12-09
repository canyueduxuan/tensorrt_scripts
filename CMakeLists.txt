cmake_minimum_required(VERSION 3.0.2)
project(tensorrt_infer)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tensorrt_models tensorrt_models)

find_package(OpenCV REQUIRED)

add_executable(crestereo_test crestereo_test.cpp)

target_link_libraries(crestereo_test
${OpenCV_LIBS}
tensorrt_models
cudart
nvinfer)

add_executable(lightstereo_test lightstereo_test.cpp)

target_link_libraries(lightstereo_test
${OpenCV_LIBS}
tensorrt_models
cudart
nvinfer)