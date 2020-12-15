add_library(Catch INTERFACE)
target_sources(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/catch)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Univariate
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_executable(tests_univariate
        univariate/distances/tests.cpp
        univariate/distances/tests_tools.cpp univariate/distances/tests_tools.hpp
        univariate/distances/dtw.cpp univariate/distances/references/dtw/dtw.cpp
        univariate/distances/cdtw.cpp univariate/distances/references/dtw/cdtw.cpp
        univariate/distances/wdtw.cpp univariate/distances/references/dtw/wdtw.cpp
        univariate/distances/elementwise.cpp
        univariate/distances/lcss.cpp univariate/distances/references/lcss/lcss.cpp
        )

target_include_directories(tests_univariate PRIVATE univariate/references)

target_link_libraries(tests_univariate PRIVATE project_options)
target_link_libraries(tests_univariate PRIVATE Catch)
target_link_libraries(tests_univariate PRIVATE tempo)
