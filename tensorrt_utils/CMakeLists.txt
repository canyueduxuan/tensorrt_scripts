cmake_minimum_required(VERSION 3.10)
project(tensorrt_utils)

file(GLOB TENSORRT_UTILS_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

find_package(CUDAToolkit REQUIRED)

add_library(tensorrt_utils STATIC
    ${TENSORRT_UTILS_SRC}
)

target_include_directories(tensorrt_utils
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(tensorrt_utils
    PUBLIC 
        nvinfer
        nvonnxparser
        nvinfer_plugin
)