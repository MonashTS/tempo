### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# CMake configuration
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
cmake_minimum_required(VERSION 3.17)
include(cmake/basis.cmake)
include(cmake/read_version.cmake)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# C++ and Interface target
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
set(CMAKE_CXX_STANDARD 17)

# Use an INTERFACE library to add option to target.
# To be used with target_link_library(target PRIVATE project_option)
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_17)
target_compile_options(project_options INTERFACE -Wall -Wextra)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Project configuration
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Read version from file
read_version(VERSION include/tempo/utils/version.hpp)
project(tempo VERSION ${VERSION})

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Creating the library
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_library(tempo INTERFACE)
target_include_directories(tempo INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_sources(tempo INTERFACE
        include/tempo/univariate/elastic_distances/distances.hpp
        include/tempo/univariate/elastic_distances/dtw/dtw.hpp
        include/tempo/utils/utils.hpp
        include/tempo/utils/version.hpp
        )

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Adding the tests
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_subdirectory(tests)