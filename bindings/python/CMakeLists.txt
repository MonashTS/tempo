### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# CMake configuration
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
cmake_minimum_required(VERSION 3.17)



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Project configuration
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
project(pytempo VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_VERBOSE_MAKEFILE ON)

# Debug build: Activate GLIBC assertions
add_compile_definitions($<$<CONFIG:DEBUG>:_GLIBCXX_ASSERTIONS>)


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Python binding
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Note: variable to be configure before calling add_subdirectory
set(PYBIND11_CPP_STANDARD -std=c++17)
add_subdirectory(lib/pybind11-2.6.1)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../ cpp EXCLUDE_FROM_ALL)

set(PYTHON_BINDINGS
        src/tempo/univariate/distances/utils.hpp
        src/tempo/univariate/distances/pybind_dtw.hpp
        src/tempo/univariate/distances/pybind_erp.hpp
        src/tempo/univariate/distances/pybind_lcss.hpp
        src/tempo/univariate/distances/pybind_msm.hpp
        src/tempo/univariate/distances/pybind_squaredED.hpp
        src/tempo/univariate/distances/pybind_twe.hpp
        src/tempo/univariate/distances/bindings.cpp
        )

# We create the submodule "distances" in the python "tempo" module
pybind11_add_module(pytempo ${PYTHON_BINDINGS})
target_link_libraries(pytempo PRIVATE tempo)