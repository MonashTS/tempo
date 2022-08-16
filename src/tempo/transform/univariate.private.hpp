#pragma once

#include <cstddef>

#include "core/univariate.derivative.hpp"

namespace tempo::transform::univariate {

  template<typename F>
  void derive(F const* data, size_t length, F* output){
    tempo::transform::derive<F, F const*, F*>(data, length, output);
  }

} // End of namespace tempo::transform::univariate
