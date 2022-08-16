#pragma once

#include <algorithm>
#include <concepts>

namespace tempo::transform {

  /** Computation of a series derivative according to "Derivative Dynamic Time Warping" by Keogh & Pazzani
   * @tparam T            Input series, must be in
   * @param series        Pointer to the series's data
   * @param length        Length of the series
   * @param out           Pointer where to write the derivative. Must be able to store 'length' values.
   * Warning: series and out should not overlap (i.e. no in-place derivation)
   */
  template<std::floating_point F, std::random_access_iterator Input, std::output_iterator<F> Output>
  void derive(Input const& series, size_t length, Output& out) {
    if (length>2) {
      for (size_t i{1}; i<length - 1; ++i) {
        out[i] = ((series[i] - series[i - 1]) + ((series[i + 1] - series[i - 1])/2.0))/2.0;
      }
      out[0] = out[1];
      out[length - 1] = out[length - 2];
    } else {
      std::copy(series, series + length, out);
    }
  }

}