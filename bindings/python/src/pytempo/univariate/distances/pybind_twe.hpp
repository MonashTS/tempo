#pragma once

#include <tempo/univariate/distances/twe/twe.hpp>
namespace cpp = tempo::univariate;

#include "../../utils.hpp"

namespace pytempo::univariate {


    // --- --- --- --- --- ---
    // TWE
    // --- --- --- --- --- ---

    inline double twe(nparray series1, nparray series2, double nu, double lambda) {
        check(series1, series2);
        return cpp::twe(USE(series1), USE(series2), nu, lambda);
    }

    inline double twe_ea(nparray series1, nparray series2, double nu, double lambda, double cutoff) {
        check(series1, series2);
        return cpp::twe(USE(series1), USE(series2), nu, lambda, cutoff);
    }


    // --- --- --- --- --- ---
    // Python module
    // --- --- --- --- --- ---

    inline void init_twe(py::module &m) {
        m.def("twe", &twe,
              "TWE between two series.",
              "serie1"_a, "serie2"_a, "nu"_a, "lambda"_a
        );

        m.def("twe", &twe_ea,
              "Early Abandoned TWE between two series. With pruning & early abandoning cut-off.",
              "serie1"_a, "serie2"_a, "nu"_a, "lambda"_a, "cutoff"_a
        );
    }

}