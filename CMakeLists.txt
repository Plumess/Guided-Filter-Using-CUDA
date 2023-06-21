cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(GuidedFilter LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

file(GLOB CU_SOURCES ./*.cu)
file(GLOB HEADERS ./*.h ./*.cuh)

add_executable(GuidedFilter ${CU_SOURCES} ${CXX_SOURCES} ${HEADERS})
set_target_properties(GuidedFilter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 设置 OpenCV 路径
set(OpenCV_DIR "E:/OpenCV/install") # 根据你的实际情况修改此处路径
find_package(OpenCV REQUIRED)
target_link_libraries(GuidedFilter ${OpenCV_LIBS})

set(CMAKE_BUILD_TYPE Debug CACHE STRING "set build type to debug")

