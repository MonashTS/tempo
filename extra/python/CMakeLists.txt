cmake_minimum_required(VERSION 3.22)
project(pytempo)


# Get PYBIND library
add_subdirectory(external/pybind11)


# Create the python bindings
pybind11_add_module(pytempo)
target_link_libraries(pytempo PRIVATE tempo)
target_sources(pytempo
        PRIVATE
        src/binder_common.hpp
        src/binder.hpp
        src/binder.cpp
        src/univariate_distance.cpp
        src/univariate_transform.cpp
        )
