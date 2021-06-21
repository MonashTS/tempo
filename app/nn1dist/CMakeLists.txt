### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# CMake configuration
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
cmake_minimum_required(VERSION 3.16)
include(../../cmake/basis.cmake)
include(../../cmake/read_version.cmake)
set(CMAKE_VERBOSE_MAKEFILE ON)


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Project configuration
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Read version from file
set(CMAKE_CXX_STANDARD 17)
read_version(VERSION ../../include/tempo/utils/version.hpp)
project(nn1dist LANGUAGES CXX VERSION ${VERSION})

add_subdirectory(../../ cpp EXCLUDE_FROM_ALL)


add_executable(nn1dist
        src/main.cpp
        src/any.cpp src/any.hpp
        ../utils/parsarg.hpp)

target_link_libraries(nn1dist PRIVATE tempo)
target_include_directories(nn1dist PRIVATE src)
