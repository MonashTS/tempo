include(FetchContent)

message(STATUS "Fetching Catch2...")

FetchContent_Declare(
        catch2
        URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.1.0.zip
)

FetchContent_MakeAvailable(catch2)
