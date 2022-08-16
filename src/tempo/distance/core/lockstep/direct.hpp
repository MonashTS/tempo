#pragma once

#include "../utils.private.hpp"

namespace tempo::distance {

  /** Direct Alignment cost, Early Abandoned.
   * @tparam F            Floating type used for the computation
   * @param length1       Length of the first series.
   * @param length2       Length of the second series.
   * @param cfun          Cost function between two points
   * @param cutoff        EAP cutoff; Attempt to prune alignments with cost > cutoff. May lead to early abandoning.
   *                      ub = PINF or QNAN: no early abandoning (direct alignment has nothing to pruned!)
   *                      ub = other value: use for pruning and early abandoning
   * @return Direct Alignment cost, or +INF if early abandoned.
   */
  template<typename F>
  F directa(const size_t length1, const size_t length2, utils::ICFun<F> auto cfun, F cutoff) {
    constexpr F PINF = utils::PINF<F>;

    if (length1!=length2) { return PINF; }
    else if (length1==0) { return 0; }
    else if (std::isinf(cutoff)||std::isnan(cutoff)) {
      // No early abandoning
      F cost = 0.0;
      for (size_t i{0}; i<length1; ++i) { cost += cfun(i, i); }
      return cost;
    } else {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds.
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      // Adjust the lower bound, taking the last alignment into account
      const F lastA = cfun(length1 - 1, length1 - 1);
      const F ub = std::nextafter(cutoff, PINF) - lastA;
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Compute the direct alignment cost, up to, excluding, the last one (already computed)
      // WARNING: We could simply start with cost = lastA instead of changing the cutoff.
      //          BUT that would change the order of floating point operation, eventually changing the result.
      //          This is usually not a problem, but we want to early abandon without ANY change on the result.
      double cost = 0;
      for (size_t i{0}; i<length1 - 1; ++i) {
        cost += cfun(i, i);
        if (cost>ub) { return PINF; }
      }
      // Add the last alignment and check the result against the original cutoff (and not 'ub')
      cost += lastA;
      if (cost>cutoff) { return PINF; } else { return cost; }
    }
  }

} // End of namespace tempo::distance
