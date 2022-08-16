#pragma once

#include "../utils.private.hpp"

namespace tempo::distance {

  /** Longest Common SubSequence (LCSS), Early Abandoned and Pruned (EAP).
   *
   * @tparam F          Floating type used for the computation
   * @param length1     Length of the first series
   * @param length2     Length of the second series
   * @param cfun_sim    Indexed Similarity function CFun<bool>
   * @param window      Warping window length - align point within 'w' (look 'w' cells on each side of the diagonal)
   *                    Having w > L-2 is the same as having no window (where L = max length)
   * @param cutoff      EAP cutoff; Attempt to prune alignments with cost > cutoff. May lead to early abandoning.
   *                    ub = PINF: Pruning using the cost of an arbitrary alignment
   *                    ub = QNAN: No cutoff: no pruning nor early abandoning
   *                    ub = other value: use for pruning and early abandoning
   * @param buffer_v    Buffer used to perform the computation. Will reallocate if required.
   * @return LCSS dissimilarity measure [0,1] where 0 stands for identical series and 1 completely distinct.
   *         +INF id early abandoned or no alignment is possible given the window w and the length of the series.
   */
  template<typename F>
  F lcss(const size_t length1,
         const size_t length2,
         utils::ICFun<bool> auto cfun_sim,
         const size_t w,
         F cutoff,
         std::vector<size_t>& buffer_v
  ) {
    constexpr F PINF = utils::PINF<F>;
    if (length1==0&&length2==0) { return 0; }
    else if ((length1==0)!=(length2==0)) { return PINF; }
    else {
      // Check that the window allows for an alignment
      // If this is accepted, we do not need to check the window when computing a new UB
      const auto m = std::min(length1, length2);
      const auto M = std::max(length1, length2);
      if (M - m>w) { return PINF; }
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Double buffer allocation, init to 0.
      // Base indices for the 'c'urrent row and the 'p'revious row. Account for the extra cell (+1 and +2)
      // Initialisation OK as is: the border on the line and "first diag" init to 0
      buffer_v.assign((1 + length2)*2, 0);
      size_t *buffers = buffer_v.data();
      size_t c{0 + 1}, p{length2 + 2};
      // Do we need to EA?
      if (cutoff>1||std::isnan(cutoff)||std::isinf(cutoff)) { // Explicitly catch inf
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // NO EA
        for (size_t i{0}; i<length1; ++i) {
          // --- --- --- Swap and variables init
          std::swap(c, p);
          const size_t jStart = utils::cap_start_index_to_window(i, w);
          const size_t jStop = utils::cap_stop_index_to_window_or_end(i, w, length2);
          // --- --- --- Init the border (very first column)
          buffers[c + jStart - 1] = 0;
          // --- --- --- Iterate through the columns
          for (size_t j{jStart}; j<jStop; ++j) {
            if (cfun_sim(i, j)) { buffers[c + j] = buffers[p + j - 1] + 1; } // Diag + 1
            else { // Note: Diagonal lookup required, e.g. when w=0
              buffers[c + j] = utils::max(buffers[c + j - 1], buffers[p + j - 1], buffers[p + j]);
            }
          } // End for loop j
        } // End for loop i
      } else {
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // WITH EA
        cutoff = std::max<F>(0.0, cutoff);
        const size_t to_reach = std::ceil((1 - cutoff)*m); // min value here
        size_t current_max = 0;
        for (size_t i{0}; i<length1; ++i) {
          // --- --- --- Stop if not enough remaining lines to reach the target (by taking the diagonal)
          const size_t lines_left = length1 - i;
          if (current_max + lines_left<to_reach) { return PINF; }
          // --- --- --- Swap and variables init
          std::swap(c, p);
          const size_t jStart = utils::cap_start_index_to_window(i, w);
          const size_t jStop = utils::cap_stop_index_to_window_or_end(i, w, length2);
          // --- --- --- Init the border (very first column)
          buffers[c + jStart - 1] = 0;
          // --- --- --- Iterate through the columns
          for (size_t j{jStart}; j<jStop; ++j) {
            if (cfun_sim(i, j)) { // Diag + 1
              const size_t cost = buffers[p + j - 1] + 1;
              current_max = std::max(current_max, cost);
              buffers[c + j] = cost;
            } // Note: Diagonal lookup required, e.g. when w=0
            else { buffers[c + j] = utils::max(buffers[c + j - 1], buffers[p + j - 1], buffers[p + j]); }
          } // End for loop j
        } // End for loop i
      } // End EA
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Finalisation: put the result on a [0 - 1] range - normalize by the minimum value (max +1 we can do)
      return 1.0 - (F(buffers[c + length2 - 1])/(F)m);
    }
  }

  /// Helper without having to provide a buffer
  template<typename F>
  inline F lcss(size_t length1, size_t length2, utils::ICFun<bool> auto cfun_sim, const size_t w, F cutoff) {
    std::vector<size_t> v;
    return lcss<F>(length1, length2, cfun_sim, w, cutoff, v);
  }


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Specific cost functions
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  namespace univariate {

    /// Parameterized Indexed Cost function Builder - return a Cost function builder
    /// Check if two floating point values are within epsilon
    /// Use by, e.g., LCSS
    template<std::floating_point F, utils::Subscriptable D>
    inline auto idx_simdiff(F e) {
      return [e](D const& lines, D const& cols) -> utils::ICFun<bool> auto {
        return [&, e](size_t i, size_t j) {
          return ad1<F>(lines[i], cols[j])<e;
        };
      };
    }

  } // End of namespace univariate

  namespace multivariate {

  } // End of namespace multivariate

} // End of namespace tempo::distance




