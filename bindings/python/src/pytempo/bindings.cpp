#include <pybind11/pybind11.h>

#include "tseries/bindings.hpp"
#include "univariate/distances/bindings.hpp"
#include "univariate/transforms/bindings.hpp"

PYBIND11_MODULE(pytempo, m) {

    // --- --- --- Add Time Series
    pytempo::submod_tseries(m);

    // --- --- --- Add Univariate
    // Create a submodule
    auto mod_univariate = m.def_submodule("univariate");
    // Add univariate distances
    pytempo::univariate::submod_distances(mod_univariate);
    // Add univariate transforms
    pytempo::univariate::submod_transforms(mod_univariate);

}


