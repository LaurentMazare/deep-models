cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cifar10-fast-cpp)

find_package(Torch REQUIRED)

add_executable(infer infer.cpp)
target_link_libraries(infer "${TORCH_LIBRARIES}")
set_property(TARGET infer PROPERTY CXX_STANDARD 11)