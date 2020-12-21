#include "pybind_dtw.hpp"
#include "pybind_erp.hpp"
#include "pybind_lcss.hpp"
#include "pybind_msm.hpp"
#include "pybind_squaredED.hpp"
#include "pybind_twe.hpp"

PYBIND11_MODULE(distances, m) {
    m.attr("__name__") = "tempo.univariate.distances";
    python::tempo::init_dtw(m);
    python::tempo::init_erp(m);
    python::tempo::init_lcss(m);
    python::tempo::init_msm(m);
    python::tempo::init_squaredED(m);
    python::tempo::init_twe(m);
}