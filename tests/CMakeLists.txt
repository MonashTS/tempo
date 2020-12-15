add_library(Catch INTERFACE)
target_sources(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/catch)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Univariate
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_executable(tests_univariate
        univariate/tests.cpp
        univariate/tests_tools.cpp univariate/tests_tools.hpp
        univariate/dtw.cpp univariate/references/dtw/dtw.cpp
        univariate/cdtw.cpp univariate/references/dtw/cdtw.cpp
        univariate/wdtw.cpp univariate/references/dtw/wdtw.cpp
        univariate/elementwise.cpp
        univariate/lcss.cpp univariate/references/lcss/lcss.cpp
        )

target_include_directories(tests_univariate PRIVATE univariate/references)

target_link_libraries(tests_univariate PRIVATE project_options)
target_link_libraries(tests_univariate PRIVATE Catch)
target_link_libraries(tests_univariate PRIVATE tempo)
