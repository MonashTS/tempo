#include "binder_common.hpp"

#include "tempo/transform/univariate.hpp"

namespace univariate_transform {

  nparray_t derive(nparray input) {
    check_univariate(input);
    nparray_t array(input.size());
    tempo::transform::univariate::derive(input.data(), input.size(), array.mutable_data());
    return std::move(array);
  }

  void init(py::module& m) {
    auto mod_transform = m.def_submodule("transform");
    mod_transform.def("derive", &derive, R"pbdoc( Compute the derivative of a univariate series)pbdoc", "series"_a);
  }

} // End of namespace univariate_transform {