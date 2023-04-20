# =================================================================================
# CMake Utilities
# =================================================================================


# Function to print all cmake variables.
# An optional regular expression is supported as a filter
# dump_cmake_variables([regex_filter])
#
# Reference:
# https://stackoverflow.com/questions/9298278/cmake-print-out-all-accessible-variables-in-a-script
#
function(dump_cmake_variables)
    get_cmake_property(_variableNames VARIABLES)
    list (SORT _variableNames)
    foreach (_variableName ${_variableNames})
        if (ARGV0)
            unset(MATCHED)
            string(REGEX MATCH ${ARGV0} MATCHED ${_variableName})
            if (NOT MATCHED)
                continue()
            endif()
        endif()
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
endfunction()

function(print_env_info)
    message(STATUS "System: ${CMAKE_SYSTEM_NAME} (${CMAKE_SYSTEM})")
    message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
    message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "Compiler TARGET: ${CMAKE_CXX_COMPILER_TARGET}")
    message(STATUS "Compiler VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
endfunction()

function(detect_os)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR (NOT CMAKE_SYSTEM_NAME AND CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux"))
        set(LINUX linux CACHE INTERNAL "OS flag")
        set(OS_NAME linux CACHE INTERNAL "OS name")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR (NOT CMAKE_SYSTEM_NAME AND CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin"))
        set(OSX osx CACHE INTERNAL "OS flag")
        set(OS_NAME osx CACHE INTERNAL "OS name")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR (NOT CMAKE_SYSTEM_NAME AND CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows"))
        set(WINDOWS windows CACHE INTERNAL "OS flag")
        set(OS_NAME windows CACHE INTERNAL "OS name")
    endif()
    message(STATUS "Detected OS: ${OS_NAME}")
endfunction()
