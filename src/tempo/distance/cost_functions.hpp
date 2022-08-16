#pragma once

#include "utils.hpp"
#include <cmath>

namespace tempo::distance {

  namespace univariate {

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Cost function
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Cost function - Absolute Difference
    template<std::floating_point F>
    inline F ad1(F a, F b) {
      return std::abs(a - b);
    }

    /// Cost function - Absolute Difference Squared
    template<std::floating_point F>
    inline F ad2(F a, F b) {
      F d = a - b;
      return d*d;
    }

    /// Parameterized Cost function -  Absolute Difference with an arbitrary exponent e
    /// Warning: up to 5 times slower than idx_ad1 or idx_ad2!
    template<std::floating_point F>
    inline F ade(F a, F b, F e) {
      return std::pow(std::abs(a - b), e);
    }

    /// Parameterized Cost function Builder -  Absolute Difference with an arbitrary exponent e
    template<std::floating_point F>
    inline utils::CFun<F> auto ade(F e) {
      return [e](F a, F b) -> F { return ade(a, b, e); };
    }


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Indexed Cost function builder
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Indexed Cost function builder - Absolute Difference exponent 1
    template<std::floating_point F, utils::Subscriptable D>
    inline utils::ICFun<F> auto idx_ad1(D const& lines, D const& cols) {
      return [&](size_t i, size_t j) {
        return ad1<F>(lines[i], cols[j]);
      };
    }

    /// Indexed Cost function builder - Absolute Difference exponent 2
    template<std::floating_point F, utils::Subscriptable D>
    inline utils::ICFun<F> auto idx_ad2(D const& lines, D const& cols) {
      return [&](size_t i, size_t j) {
        return ad2<F>(lines[i], cols[j]);
      };
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Parameterized Indexed Cost function builder
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Parameterized Indexed Cost function builder - return a Cost function builder
    /// Absolute Difference with an arbitrary exponent e
    template<std::floating_point F, utils::Subscriptable D>
    inline auto idx_ade(F e) {
      return [e](D const& lines, D const& cols) -> utils::ICFun<F> auto {
        return [&, e](size_t i, size_t j) {
          return ade<F>(lines[i], cols[j], e);
        };
      };
    }

  } // End of namespace univariate

  namespace multivariate {

  } // End of namespace multivariate

} // End of namespace tempo::distance
