#include "univariate.private.hpp"

namespace tempo::transform::univariate {

  // Implementation through template explicit instantiation

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Double implementation
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  using F = double;

  template void derive<F>(F const* data, size_t length, F* output);

  /*
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Float implementation
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  using Ff = float;

  template void derive<Ff>(Ff const* data, size_t length, Ff* output);
  */

} // End of namespace tempo::transform::univariate
