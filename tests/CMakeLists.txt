add_library(Catch INTERFACE)
target_sources(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/catch)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Univariate
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_executable(tests
        # --- --- --- Entry point
        tests.cpp
        # --- --- --- Univariate
        # --- Tools
        univariate/tests_tools.cpp univariate/tests_tools.hpp
        # --- Classifiers
        univariate/classifiers/nn1.cpp
        # --- Distances
        univariate/distances/dtw.cpp univariate/distances/references/dtw/dtw.cpp
        univariate/distances/cdtw.cpp univariate/distances/references/dtw/cdtw.cpp
        univariate/distances/wdtw.cpp univariate/distances/references/dtw/wdtw.cpp
        univariate/distances/elementwise.cpp
        univariate/distances/erp.cpp univariate/distances/references/erp/erp.cpp
        univariate/distances/lcss.cpp univariate/distances/references/lcss/lcss.cpp
        univariate/distances/msm.cpp univariate/distances/references/msm/msm.cpp
        univariate/distances/twe.cpp univariate/distances/references/twe/twe.cpp
        )

target_include_directories(tests PRIVATE univariate/references)

target_link_libraries(tests PRIVATE project_options)
target_link_libraries(tests PRIVATE Catch)
target_link_libraries(tests PRIVATE tempo)
