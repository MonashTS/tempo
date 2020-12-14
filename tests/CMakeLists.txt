add_library(Catch INTERFACE)
target_sources(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/catch)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Univariate
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
add_executable(tests_univariate
        univariate/tests.cpp
        )

target_include_directories(tests_univariate PRIVATE univariate/references)

target_link_libraries(tests_univariate PRIVATE project_options)
target_link_libraries(tests_univariate PRIVATE Catch)
target_link_libraries(tests_univariate PRIVATE tempo)
