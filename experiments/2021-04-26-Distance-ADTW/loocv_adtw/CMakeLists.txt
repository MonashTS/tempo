### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# CMake configuration
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
cmake_minimum_required(VERSION 3.17)
set(CMAKE_VERBOSE_MAKEFILE ON)


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Project configuration
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Read version from file
set(CMAKE_CXX_STANDARD 20)
project(exp_sed LANGUAGES CXX)

add_subdirectory(../../../ cpp EXCLUDE_FROM_ALL)


add_executable(loocv_adtw src/main.cpp)

target_link_libraries(loocv_adtw PRIVATE tempo)
target_include_directories(loocv_adtw PRIVATE src)
