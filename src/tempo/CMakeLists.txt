# Create the libtempo target
add_library(libtempo)
target_include_directories(libtempo PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..)

# Gather the sources (including the header)
target_sources(libtempo PRIVATE predef.hpp )

# Recurse (also run for the tests)
add_subdirectory(utils)

add_subdirectory(classifier)
add_subdirectory(dataset)
add_subdirectory(distance)
add_subdirectory(transform)

# Linking
target_link_libraries(libtempo PUBLIC ${MLPACK_LIBRARY} ${ARMADILLO_LIBRARIES} rapidcsv json)

if(OpenMP_CXX_FOUND)
    target_link_libraries(libtempo PUBLIC OpenMP::OpenMP_CXX)
endif()