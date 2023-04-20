target_sources(libtempo
        # --- --- ---
        # --- --- ---
        # --- --- ---
        PUBLIC
        # General interface
        nn1splitter.hpp
        nn1dist_interface.hpp
        nn1dist_base.hpp
        # Specific implementation
        nn1_directa.hpp
        nn1_adtw.hpp
        nn1_dtw.hpp
        nn1_dtwfull.hpp
        nn1_wdtw.hpp
        nn1_erp.hpp
        nn1_lcss.hpp
        nn1_msm.hpp
        nn1_twe.hpp
        # --- --- ---
        # --- --- ---
        # --- --- ---
        PRIVATE
        nn1splitter.private.hpp
        nn1splitter.cpp
        nn1_adtw.cpp
        nn1_directa.cpp
        nn1_dtw.cpp
        nn1_dtwfull.cpp
        nn1_erp.cpp
        nn1_lcss.cpp
        nn1_msm.cpp
        nn1_twe.cpp
        nn1_wdtw.cpp
)