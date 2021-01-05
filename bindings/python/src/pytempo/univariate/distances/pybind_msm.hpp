#pragma once

#include <tempo/univariate/distances/msm/msm.hpp>
namespace cpp = tempo::univariate;

#include "../../utils.hpp"

namespace pytempo::univariate {


    // --- --- --- --- --- ---
    // MSM
    // --- --- --- --- --- ---

    inline double msm(nparray series1, nparray series2, double cost) {
        check(series1, series2);
        return cpp::msm(series1.data(), series1.size(), series2.data(), series2.size(), cost);
    }

    inline double msm_ea(nparray series1, nparray series2, double cost, double cutoff) {
        check(series1, series2);
        return cpp::msm(series1.data(), series1.size(), series2.data(), series2.size(), cost, cutoff);
    }


    // --- --- --- --- --- ---
    // Python module
    // --- --- --- --- --- ---

    inline void init_msm(py::module &m) {
        m.def("msm", &msm,
              "MSM between two series.",
              "serie1"_a, "serie2"_a, "cost"_a);

        m.def("msm_ea", &msm_ea,
              "MSM between two series. With pruning & early abandoning cut-off.",
              "serie1"_a, "serie2"_a, "cost"_a, "cutoff"_a
        );

    }
}