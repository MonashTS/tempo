#pragma once

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#define USE(series) series.data(), series.size()

namespace pytempo {

    /// Alias for pybind11 namespace
    namespace py = pybind11;

    /// Alias for numpy array: must be made of double, dense c style, with conversion (copy to dense) if not.
    using nparray_t = py::array_t<double, py::array::c_style | py::array::forcecast>;

    /// Alias for reference constant numpy array
    using nparray = const nparray_t&;

    /// Alias for mutable numpy array
    using nparray_mut = nparray_t&;

    /// Using literals
    using namespace pybind11::literals;

    /// Check if a numpy array has a given dimension, 1 by default.
    inline void check_dimension(nparray series1, ssize_t nbdim=1) {
        if (series1.ndim() != nbdim) {
            throw std::invalid_argument("Multivariate not implemented");
        }
    }

    /// Check if two numpy arrays have the same dimension
    inline void check_same_dimension(nparray series1, nparray series2) {
        if (series1.ndim() != series2.ndim()) {
            throw std::invalid_argument("Series are of different dimensions");
        }
    }

    /// Check series
    inline void check(nparray series1, nparray series2){
        check_same_dimension(series1, series2);
        check_dimension(series1, 1);
    }

    /// Check if two numpy arrays have the same length
    inline void check_same_length(nparray a1, nparray a2) {
      if (a1.size() != a2.size()) {
        throw std::invalid_argument("Length are not matching");
      }
    }


}