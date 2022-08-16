#include "binder.hpp"

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Putting the module together

PYBIND11_MODULE(pytempo, m) {

  // --- Univariate
  auto mod_univariate = m.def_submodule("univariate");
  // --- --- distances
  univariate_distance::init(mod_univariate);
  // --- --- transform
  univariate_transform::init(mod_univariate);

}