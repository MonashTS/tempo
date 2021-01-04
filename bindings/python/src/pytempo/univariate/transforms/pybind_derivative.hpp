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
        std::vector<ssize_t> shape (series1.shape(), series1.shape()+series1.ndim());
        out.resize(shape, false);
        cpp::derivative(USE(series1), out.mutable_data());
    }

    // --- --- --- --- --- ---
    // Python module
    // --- --- --- --- --- ---

    inline void init_derivative(py::module &m) {
        m.def("derivative", &derivative,
              "Return the derivative of the series in a newly allocated array.",
              "series"_a
        );

        const auto* txt =
                "Return the derivative of the series in the out array (resized if necessary)."
                "\nAllow to reuse an array, possibly saving costly allocation."
                "\nBest used when no resizing is required."
                "\nWarning: do not keep references on 'out': resizing will invalidate them!";

        m.def("derivative", &derivative_out, txt, "series"_a, "out"_a );

    }
}