cmake_minimum_required(VERSION 3.0.2)
project(tensorrt_models)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../tensorrt_utils tensorrt_utils)

file(GLOB TENSORRT_MODELS_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(tensorrt_models STATIC
    ${TENSORRT_MODELS_SRC}
)

target_include_directories(tensorrt_models
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
tensorrt_utils
cudart
)