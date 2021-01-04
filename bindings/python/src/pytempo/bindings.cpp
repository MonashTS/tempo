#include <pybind11/pybind11.h>

#include "utils.hpp"
#include "univariate/distances/bindings.hpp"

PYBIND11_MODULE(pytempo, m) {
    // --- --- --- Add univariate module
    auto mod_univariate = m.def_submodule("univariate");
    // Add univariate distances
    pytempo::univariate::submod_distances(mod_univariate);

}


