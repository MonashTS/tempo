#pragma once

#include <cstddef>

namespace tempo::transform::univariate {

  template<typename F>
  void derive(F const* data, size_t length, F* output);

} // End of namespace tempo::transform::univariate
