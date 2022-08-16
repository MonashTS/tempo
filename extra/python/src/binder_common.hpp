#pragma once

/// Our floating point type
using F = double;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Pybind11 & associated tooling

#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

/// Alias for numpy array: must be made of F, dense c style, with conversion (copy to dense) if not.
using nparray_t = py::array_t<F, py::array::c_style | py::array::forcecast>;

/// Alias for reference constant numpy array
using nparray = const nparray_t&;

/// Alias for mutable numpy array
using nparray_mut = nparray_t&;

/// Check if a numpy array has a given dimension, 1 by default.
inline void check_dimension(nparray series1, int ndim = 1) {
  if (series1.ndim()!=ndim) { throw std::invalid_argument("Multivariate not implemented"); }
}

/// Check for two univariate arrays
inline void check_univariate(nparray series1) {
  check_dimension(series1, 1);
}

/// Check for two univariate arrays
inline void check_univariate(nparray series1, nparray series2) {
  check_dimension(series1, 1);
  check_dimension(series2, 1);
}

/// Check if two numpy arrays have the same length
inline void check_same_length(nparray a1, nparray a2) {
  if (a1.size()!=a2.size()) { throw std::invalid_argument("Length are not matching"); }
}

/// Helper macro for using nparray (see distance function below)
#define USE(series) (series).data(), (series).size()
