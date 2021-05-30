cmake_minimum_required(VERSION 2.8.12)
project(tp2)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

message("Wasm mode")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -s ALLOW_MEMORY_GROWTH=1 --bind")

include_directories(../eigen)

message(CMAKE_INSTALL_PREFIX = "${CMAKE_INSTALL_PREFIX}")

add_executable(tp2
        ../src/main_embind.cpp
        ../src/main.cpp
        ../src/knn.cpp
        ../src/pca.cpp
        ../src/eigen.cpp)