#pragma once

#include <tempo/univariate/transforms/derivative.hpp>

#include "pybind_derivative.hpp"

namespace pytempo::univariate {

    /// Add a submodule named "transforms" into "m"
    inline void submod_transforms(py::module& m){
        auto sm = m.def_submodule("transforms");
        init_derivative(sm);
    }

}
