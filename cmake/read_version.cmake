function(read_version RESULT file_name)
    message(STATUS "Reading version from ${file_name}")

    # Assuming the canonical version is listed in a single line
    # This would be in several parts if picking up from MAJOR, MINOR, etc.
    set(VERSION_REGEX "#define PROJECT_VERSION[ \t]+\"(.+)\"")

    # Read in the line containing the version
    #file(STRINGS ${file_name} VERSION_STRING REGEX ${VERSION_REGEX})
    file(STRINGS ${file_name} VERSION_STRING REGEX ${VERSION_REGEX})

    # Pick out just the version
    string(REGEX REPLACE ${VERSION_REGEX} "\\1" VERSION_STRING "${VERSION_STRING}")

    # Result
    set(${RESULT} ${VERSION_STRING} PARENT_SCOPE)
endfunction()