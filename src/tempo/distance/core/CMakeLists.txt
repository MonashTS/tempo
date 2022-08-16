### Distances
target_sources(tempo
        # --- --- ---
        PUBLIC
        # Elastics
        elastic/adtw.hpp
        elastic/dtw.hpp
        elastic/dtw_lb_keogh.hpp
        elastic/dtw_lb_enhanced.hpp
        elastic/dtw_lb_webb.hpp
        elastic/erp.hpp
        elastic/lcss.hpp
        elastic/msm.hpp
        elastic/twe.hpp
        elastic/wdtw.hpp
        # Lock Step
        lockstep/direct.hpp
        # --- --- ---
        PRIVATE
        utils.private.hpp
        )

### Testing
if (BUILD_TESTING)
    target_sources(tests
            # --- --- ---
            PRIVATE
            # Elastic
            elastic/adtw.test.univariate.cpp
            elastic/dtw.test.univariate.cpp
            elastic/erp.test.univariate.cpp
            elastic/lcss.test.univariate.cpp
            elastic/msm.test.univariate.cpp
            elastic/twe.test.univariate.cpp
            elastic/wdtw.test.univariate.cpp
            # Lock Step
            lockstep/direct.test.cpp
            )
endif ()