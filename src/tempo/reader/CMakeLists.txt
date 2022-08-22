target_sources(tempo
        PUBLIC
        reader_output.hpp
        csv/univariate/csv.hpp
        PRIVATE
        csv/univariate/csv.template.cpp
        csv/univariate/csv.cpp
        )


### Testing
if (BUILD_TESTING)
    target_sources(tests
            # --- --- ---
            PRIVATE
            csv/univariate/csv.test.cpp
            )
endif ()
