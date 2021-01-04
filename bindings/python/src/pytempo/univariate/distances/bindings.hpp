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
        pytempo::init_dtw(sm);
        pytempo::init_erp(sm);
        pytempo::init_lcss(sm);
        pytempo::init_msm(sm);
        pytempo::init_squaredED(sm);
        pytempo::init_twe(sm);
    }

}
