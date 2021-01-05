#pragma once

#include <tempo/univariate/distances/erp/erp.hpp>
namespace cpp = tempo::univariate;

#include "../../utils.hpp"

namespace pytempo::univariate {

    // --- --- --- --- --- ---
    // ERP
    // --- --- --- --- --- ---

    inline double erp(nparray series1, nparray series2, double gValue, size_t w) {
        check(series1, series2);
        return cpp::erp(series1.data(), series1.size(), series2.data(), series2.size(), gValue, w);
    }

    inline double erp_ea(nparray series1, nparray series2, double gValue, size_t w, double cutoff) {
        check(series1, series2);
        return cpp::erp(series1.data(), series1.size(), series2.data(), series2.size(), gValue, w, cutoff);
    }


    // --- --- --- --- --- ---
    // Python module
    // --- --- --- --- --- ---

    inline void init_erp(py::module &m) {
        m.def("erp", &erp, "ERP between two series. \"w\" is the half-window size.",
              "serie1"_a, "serie2"_a, "gValue"_a, "w"_a
        );

        m.def("erp", &erp_ea,
              "Early Abandoned ERP between two series. \"w\" is the half-window size. With pruning & early abandoning cut-off.",
              "serie1"_a, "serie2"_a, "gValue"_a, "w"_a, "cutoff"_a
        );
    }

}
