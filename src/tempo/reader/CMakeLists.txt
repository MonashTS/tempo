target_sources(tempo
        PUBLIC
        reader_output.hpp
        csv/csv.hpp
        PRIVATE
        csv/csv.template.cpp
        csv/csv.cpp
        )


### Testing
if (BUILD_TESTING)
    target_sources(tests
            # --- --- ---
            PRIVATE
            csv/csv.test.cpp
            )
endif ()
