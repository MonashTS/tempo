add_subdirectory(core)

target_sources(tempo
        # Implementation
        PUBLIC
        utils.hpp
        cost_functions.hpp
        univariate.hpp
        PRIVATE
        univariate.private.hpp
        univariate.cpp
        )
