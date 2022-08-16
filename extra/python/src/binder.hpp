#pragma once
#include "binder_common.hpp"

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Distances - see ./univariate_distance.cpp

namespace univariate_distance {
  void init(py::module& m);
}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Transform - see ./univariate_transform.cpp

namespace univariate_transform {
  void init(py::module& m);
}
