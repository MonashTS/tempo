#pragma once

#include <tempo/univariate/distances/lcss/lcss.hpp>
namespace cpp = tempo::univariate;

#include "../../utils.hpp"

namespace pytempo::univariate {


    // --- --- --- --- --- ---
    // LCSS
    // --- --- --- --- --- ---

    inline double lcss(nparray series1, nparray series2, double epsilon, size_t w) {
        check(series1, series2);
        return cpp::lcss(series1.data(), series1.size(), series2.data(), series2.size(), epsilon, w);
    }

    inline double lcss_ea(nparray series1, nparray series2, double epsilon, size_t w, double cutoff) {
        check(series1, series2);
        return cpp::lcss(series1.data(), series1.size(), series2.data(), series2.size(), epsilon, w, cutoff);
    }

    // --- --- --- --- --- ---
    // Python module
    // --- --- --- --- --- ---

    inline void init_lcss(py::module &m) {
        m.def("lcss", &lcss,
              "LCSS between two series. \"w\" is the half-window size.",
              "series1"_a, "series2"_a, "epsilon"_a, "w"_a
              );

        m.def("lcss", &lcss_ea,
              "Early Abandoned LCSS between two series. \"w\" is the half-window size. With pruning & early abandoning cut-off.",
              "series1"_a, "series2"_a, "epsilon"_a, "w"_a, "cutoff"_a
        );
    }

}
