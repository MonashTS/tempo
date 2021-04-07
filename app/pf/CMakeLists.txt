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
project(pf LANGUAGES CXX VERSION ${VERSION})

add_subdirectory(../../ cpp EXCLUDE_FROM_ALL)


add_executable(pf
        src/main.cpp
        ../utils/parsarg.hpp
        ../utils/parsarg_predefs.hpp
        src/any.hpp
        src/any.cpp
        )

target_link_libraries(pf PRIVATE tempo)
target_include_directories(pf PRIVATE src)
