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


add_executable(exp_sed src/main.cpp)

target_link_libraries(exp_sed PRIVATE tempo)
target_include_directories(exp_sed PRIVATE src)
