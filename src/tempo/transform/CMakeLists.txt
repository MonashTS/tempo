add_subdirectory(core)

target_sources(tempo
        # Implementation
        PUBLIC
        univariate.hpp
        PRIVATE
        univariate.cpp
        univariate.private.hpp
        )
