include(FetchContent)

message(STATUS "Fetching Pybind11...")

FetchContent_Declare(
        pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v2.10.0.zip
)

FetchContent_MakeAvailable(pybind11)
