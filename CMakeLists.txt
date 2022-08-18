cmake_minimum_required(VERSION 3.20)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Configurable options
option(BUILD_TESTING "Build tests." ON)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Project
project(tempo)
set(CMAKE_CXX_STANDARD 20)



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# CPP tempo

### Dependencies
# RapidCSV
add_subdirectory(external/rapidcsv-8.64)

### Distances
# Create a STATIC library with position independent code instantiating our headers
add_library(tempo STATIC)
set_target_properties(tempo PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tempo PUBLIC src)
# Dependencies
target_link_libraries(tempo PUBLIC rapidcsv)

### Testing
# All the tests are compiled into one executable.
# Recursing into src/tempo will add the tests per component to the target created here
if(BUILD_TESTING)
    # Catch2 Dependencies: fetch content!
    add_subdirectory(external/Catch2)
    add_executable(tests test/tests.cpp)
    target_include_directories(tests PRIVATE test)
    target_link_libraries(tests PRIVATE tempo Catch2::Catch2WithMain)
endif()

### Get the sources and tests
add_subdirectory(src/tempo)



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Python bindings
add_subdirectory(extra)
