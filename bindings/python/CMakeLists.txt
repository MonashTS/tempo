### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# CMake configuration
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
cmake_minimum_required(VERSION 3.19)



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Project configuration
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
project(pytempo LANGUAGES CXX VERSION 0.0.1)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)

if(MSVC)
    add_compile_options(/W3)
else()
    add_compile_options(-Wall -Wextra)
endif()


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Python binding
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Note: variable to be configure before calling add_subdirectory
set(PYBIND11_CPP_STANDARD -std=c++20)
add_subdirectory(lib/pybind11-2.9.0)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../ cpp EXCLUDE_FROM_ALL)

set(PYTHON_BINDINGS
        src/pytempo/utils.hpp
        src/pytempo/bindings.cpp
        # --- --- --- TSeries
        src/pytempo/tseries/bindings.hpp
        # --- --- --- Univariate
        # --- Distances
        src/pytempo/univariate/distances/pybind_dtw.hpp
        src/pytempo/univariate/distances/pybind_erp.hpp
        src/pytempo/univariate/distances/pybind_lcss.hpp
        src/pytempo/univariate/distances/pybind_msm.hpp
        src/pytempo/univariate/distances/pybind_squaredED.hpp
        src/pytempo/univariate/distances/pybind_twe.hpp
        src/pytempo/univariate/distances/bindings.hpp
        # --- Transforms
        src/pytempo/univariate/transforms/bindings.hpp
        src/pytempo/univariate/transforms/pybind_derivative.hpp
        )

# We create the submodule "distances" in the python "tempo" module
pybind11_add_module(pytempo ${PYTHON_BINDINGS})
target_link_libraries(pytempo PRIVATE tempo)