#pragma once

#include <tempo/univariate/distances/elementwise/elementwise.hpp>
namespace cpp = tempo::univariate;

#include "../../utils.hpp"

namespace pytempo::univariate {

    // --- --- --- --- --- ---
    // Squared Euclidean Distance
    // --- --- --- --- --- ---

    inline double squaredED(nparray series1, nparray series2) {
        check(series1, series2);
        return cpp::elementwise(USE(series1), USE(series2));
    }

    inline double squaredED_ea(nparray series1, nparray series2, double cutoff) {
        check(series1, series2);
        return cpp::elementwise(USE(series1), USE(series2), cutoff);
    }

    // --- --- --- --- --- ---
    // Python module
    // --- --- --- --- --- ---

    inline void init_squaredED(py::module &m) {
        m.def("squaredED", &squaredED,
              "Squared Euclidean Distance between two series.",
              "serie1"_a, "serie2"_a
        );

        m.def("squaredED", &squaredED_ea,
              "Squared Euclidean Distance between two series. With early abandoning cut-off.",
              "serie1"_a, "serie2"_a, "cutoff"_a
        );
    }

}