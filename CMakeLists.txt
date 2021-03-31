### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# CMake configuration
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
cmake_minimum_required(VERSION 3.17)
include(cmake/basis.cmake)
include(cmake/read_version.cmake)
set(CMAKE_VERBOSE_MAKEFILE ON)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# C++ and Interface target
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
set(CMAKE_CXX_STANDARD 17)

# Use an INTERFACE library to add option to target.
# To be used with target_link_library(target PRIVATE project_option)
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_17)
target_compile_options(project_options INTERFACE -Wall -Wextra)
target_compile_definitions(project_options INTERFACE $<$<CONFIG:DEBUG>:_GLIBCXX_ASSERTIONS>)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Project configuration
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Read version from file
read_version(VERSION include/tempo/utils/version.hpp)
project(tempo VERSION ${VERSION})

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Creating the library
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
set(TEMPO_SRC
        # --- --- --- Readers
        src/tempo/reader/readingtools.hpp
        include/tempo/reader/ts/ts.hpp
        src/tempo/reader/ts/ts.cpp
        # --- --- --- TSeries
        include/tempo/tseries/dataset.hpp
        include/tempo/tseries/indexSet.hpp
        include/tempo/tseries/transform.hpp
        include/tempo/tseries/tseries.hpp
        # --- --- --- Univariate
        # --- --- --- --- Classifiers
        # --- --- --- --- --- NN1
        include/tempo/univariate/classifiers/nn1/nn1.hpp
        # --- --- --- --- --- PF
        include/tempo/univariate/classifiers/proximity_forest/pf.hpp
        include/tempo/univariate/classifiers/proximity_forest/splitters/splitters.hpp
        include/tempo/univariate/classifiers/proximity_forest/splitters/distances_splitters.hpp
        # --- --- --- --- Distances
        include/tempo/univariate/distances/distances.hpp
        include/tempo/univariate/distances/dtw/lowerbound/envelopes.hpp
        include/tempo/univariate/distances/dtw/lowerbound/lb_enhanced.hpp
        include/tempo/univariate/distances/dtw/lowerbound/lb_keogh.hpp
        include/tempo/univariate/distances/dtw/lowerbound/lb_webb.hpp
        include/tempo/univariate/distances/dtw/dtw.hpp
        include/tempo/univariate/distances/dtw/cdtw.hpp
        include/tempo/univariate/distances/dtw/wdtw.hpp
        include/tempo/univariate/distances/elementwise/elementwise.hpp
        include/tempo/univariate/distances/erp/erp.hpp
        include/tempo/univariate/distances/lcss/lcss.hpp
        include/tempo/univariate/distances/msm/msm.hpp
        # --- --- --- --- Transforms
        include/tempo/univariate/transforms/derivative.hpp
        # --- --- --- Utils
        include/tempo/utils/utils/rand.hpp
        include/tempo/utils/utils/stats.hpp
        include/tempo/utils/utils/timing.hpp
        include/tempo/utils/utils/unsignedutils.hpp
        include/tempo/utils/utils.hpp
        include/tempo/utils/capsule.hpp
        include/tempo/utils/uncopyable.hpp
        include/tempo/utils/version.hpp
        include/tempo/utils/jsonvalue.hpp
        )

add_library(tempo SHARED ${TEMPO_SRC})
set_target_properties(tempo PROPERTIES VERSION ${VERSION})
target_include_directories(tempo PUBLIC "include")
target_include_directories(tempo PRIVATE "src")

add_executable(test_tempo ${TEMPO_SRC})
target_include_directories(test_tempo PUBLIC "include")
target_include_directories(test_tempo PRIVATE "src")

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Adding the tests
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_subdirectory(tests)