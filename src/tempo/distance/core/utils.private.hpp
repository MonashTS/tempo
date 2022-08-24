#pragma once

#include "../utils.hpp"
#include "../cost_functions.hpp"

#include <cassert>
#include <cstddef>
#include <cmath>

#include <stdexcept>
#include <vector>

namespace tempo::distance {

  namespace utils {

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Constant
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    /// Lower Bound initial negative value: allow to deal with numerical instability.
    /// The lower bound will be (really really) marginally less tight.
    template<typename F>
    const F INITLB{-pow(F(10), -(std::numeric_limits<F>::digits10 - 1))};

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Numeric
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    template<typename F>
    F min(F a, F b, F c) {
      return std::min<F>(a, std::min<F>(b, c));
    }

    template<typename F>
    F max(F a, F b, F c) {
      return std::max<F>(a, std::max<F>(b, c));
    }

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Unsigned tooling
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /// Unsigned arithmetic: Given an 'index' and a 'window',
    /// get the start index corresponding to std::max(0, index-window)
    inline size_t cap_start_index_to_window(size_t index, size_t window) {
      if (index>window) { return index - window; } else { return 0; }
    }

    /// Unsigned arithmetic:
    ///  Given an 'index', a 'window' and an 'end', get the stop index corresponding to std::min(end, index+window+1).
    ///  The expression index+window+1 is illegal for any index>0 as window could be MAX-1
    inline size_t cap_stop_index_to_window_or_end(size_t index, size_t window, size_t end) {
      // end-window is valid when window<end
      if (window<end&&index + 1<end - window) { return index + window + 1; } else { return end; }
    }

    /// Absolute value for any comparable and subtractive type, without overflowing risk for unsigned types.
    template<typename T> requires std::unsigned_integral<T>
    inline T absdiff(T a, T b) { return (a>b) ? a - b : b - a; }

    /// From unsigned to signed for integral types
    template<typename UIType>
    inline typename std::make_signed_t<UIType> to_signed(UIType ui) {
      static_assert(std::is_unsigned_v<UIType>, "Template parameter must be an unsigned type");
      using SIType = std::make_signed_t<UIType>;
      if (ui>(UIType)(std::numeric_limits<SIType>::max())) {
        throw std::overflow_error("Cannot store unsigned type in signed type.");
      }
      return (SIType)ui;
    }

  } // End of namespace utils

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Warping Result
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  namespace WR {

    /// Returns the cost along with warping path information; requires specialised implementations
    template<typename F>
    struct WarpingResult {

      /// A cost of +Infinity means "early abandoned"
      F cost;

      /// Maximum deviation from the diagonal.
      /// Also acts as Windows validity (smallest window giving the same results, other parameters being fixed)/
      size_t max_deviation;

      /// Empty result with cost initialised to +INF and max_deviation to "NO_WINDOW" (virtually infinite length)
      WarpingResult() : cost(utils::PINF<F>), max_deviation(utils::NO_WINDOW) {}

      WarpingResult(F c, size_t md) : cost(c), max_deviation(md) {}
    };

  } // End of namespace WE

} // End of namespace tempo::distance::core
