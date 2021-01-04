#pragma once

#include <tempo/univariate/transforms/derivative.hpp>
namespace cpp = tempo::univariate;

#include "../../utils.hpp"

namespace pytempo::univariate {

    // --- --- --- --- --- ---
    // Derivative
    // --- --- --- --- --- ---

    inline nparray_t derivative(nparray series1) {
        auto result = nparray_t(series1.size());
        cpp::derivative(USE(series1), result.mutable_data());
        return result;
    }

    inline void derivative_out(nparray series1, nparray_mut out) {
    }

    // --- --- --- --- --- ---
    // Python module
    // --- --- --- --- --- ---

    inline void init_derivative(py::module &m) {
        m.def("derivative", &derivative,
              "Return the derivative of the series in a newly allocated array.",
              "series"_a
        );

        m.def("derivative", &derivative_out,
              "Return the derivative of the series in the out array, resized to the length of the input."
              "series"_a, "out"_a
        );

    }
}