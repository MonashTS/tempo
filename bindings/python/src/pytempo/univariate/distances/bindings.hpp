#include "pybind_dtw.hpp"
#include "pybind_erp.hpp"
#include "pybind_lcss.hpp"
#include "pybind_msm.hpp"
#include "pybind_squaredED.hpp"
#include "pybind_twe.hpp"

namespace pytempo::univariate {

    /// Add a submodule named "distances" into "m", and register the various (univariate) distances.
    inline void submod_distances(py::module& m){
        auto sm = m.def_submodule("distances");
        init_dtw(sm);
        init_erp(sm);
        init_lcss(sm);
        init_msm(sm);
        init_squaredED(sm);
        init_twe(sm);
    }

}
