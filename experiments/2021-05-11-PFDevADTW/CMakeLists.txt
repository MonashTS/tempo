### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# CMake configuration
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
cmake_minimum_required(VERSION 3.17)
include(../../cmake/basis.cmake)
include(../../cmake/read_version.cmake)
set(CMAKE_VERBOSE_MAKEFILE ON)


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Project configuration
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Read version from file
set(CMAKE_CXX_STANDARD 17)
read_version(VERSION ../../include/tempo/utils/version.hpp)
project(pfdevADTW LANGUAGES CXX VERSION ${VERSION})

add_subdirectory(../../ cpp EXCLUDE_FROM_ALL)


add_executable(pfdevADTW
        src/main.cpp
        ../../app/utils/parsarg.hpp
        ../../app/utils/parsarg_predefs.hpp
        src/any.hpp
        src/any.cpp
        )

target_link_libraries(pfdevADTW PRIVATE tempo)
target_include_directories(pfdevADTW PRIVATE src)
