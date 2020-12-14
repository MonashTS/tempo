# Be verbose!
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_AUTOGEN_VERBOSE ON)

# Display the source and binary directories paths
message(STATUS "Source directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "Build directory:  ${CMAKE_BINARY_DIR}")

# Check the build type
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to RelWithDebInfo")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
else()
    message(STATUS "Build type configured on ${CMAKE_BUILD_TYPE}")
endif()