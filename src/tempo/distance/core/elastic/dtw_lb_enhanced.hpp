#pragma once

#include "../utils.private.hpp"

namespace tempo::distance {

  namespace univariate {

    /** LB Enhanced - only applicable for same-length series.
     * Note:  to deal with numerical instability, the computation is initialized with a negative value close to 0:
     *        It follows that the result is (really really) marginally less tight than expected,
     *        but reduces the risk of producing a value too high (due to floating point precision).
     * @tparam F                Floating type used for the computation
     * @param query             The time series being queried
     * @param query_length      Length of query
     * @param candidate         Time series from the database
     * @param candidate_length  Length of candidate
     * @param candidate_upper           Upper envelope of candidate
     * @param candidate_lower           Lower envelope of candidate
     * @param cfun            Cost Function
     * @param v               Speed/Tightness trade-off (faster = 0, tighter = min(lq, lc)/2)
     * @param w               Warping window
     * @param cutoff          Cut-off value (strictly) above which we early abandon ("best so far")
     * @return +INF if early abandoned, or the lower bound value
     */
    template<typename F>
    F lb_Enhanced(
      const F *query, size_t query_length,
      const F *candidate, [[maybe_unused]] size_t candidate_length,
      const F *candidate_upper, const F *candidate_lower,
      utils::CFun<F> auto cfun,
      size_t v,
      size_t w,
      F cutoff
    ) {
      assert(query_length==candidate_length);
      F lb{utils::INITLB<F>}; // Init with a small negative value: handle numerical instability
      size_t nbands = std::min(query_length/2, v);
      // --- --- --- Do L & R Bands
      // First alignment
      lb += cfun(query[0], candidate[0]);
      // Manage the case of series of length 1
      if (query_length==1) { return (lb>cutoff) ? utils::PINF<F> : lb; }
      const size_t last = query_length - 1;
      // Last alignment
      lb += cfun(query[last], candidate[last]);
      // L & R bands
      for (size_t i = 1; i<nbands&&lb<=cutoff; ++i) {
        const auto fixR = last - i;
        F minL = cfun(query[i], candidate[i]);
        F minR = cfun(query[fixR], candidate[fixR]);
        for (size_t j = utils::cap_start_index_to_window(i, w); j<i; ++j) {
          const auto movR = last - j;
          minL = utils::min(minL, cfun(query[i], candidate[j]), cfun(query[j], candidate[i]));
          minR = utils::min(minR, cfun(query[fixR], candidate[movR]), cfun(query[movR], candidate[fixR]));
        }
        lb = lb + minL + minR;
      }
      // --- --- ---
      if (lb>cutoff) { return utils::PINF<F>; }

      // --- --- --- Bridge with LB Keogh, continue while we are <= cutoff
      const auto end = query_length - nbands;
      for (size_t i = nbands; i<end&&lb<=cutoff; ++i) {
        F qi{query[i]};
        if (const auto ui{candidate_upper[i]}; qi>ui) { lb += cfun(qi, ui); }
        else if (const auto li{candidate_lower[i]}; qi<li) { lb += cfun(qi, li); }
      }

      return (lb>cutoff) ? utils::PINF<F> : lb;
    }

    /** LB Enhanced 2 ways 'joined' - only applicable for same-length series.
     * Combination of lb_Enhenced and lb_Keogh_2j.
     * @tparam F          Floating type used for the computation
     * @param series1     First series
     * @param length1     First series length
     * @param upper1      First series upper envelope
     * @param lower1      First series lower envelope
     * @param series2     Second series
     * @param length2     Second series length
     * @param upper2      Second series upper envelope
     * @param lower2      Second series lower envelope
     * @param cfun        Cost Function
     * @param v           Speed/Tightness trade-off (faster = 0, tighter = min(lq, lc)/2)
     * @param w           Warping window
     * @param cutoff      Cut-off value (strictly) above which we early abandon ("best so far")
     * @return +INF if early abandoned, or the lower bound value
     */
    template<typename F>
    F lb_Enhanced2j(
      const F *series1, size_t length1,
      const F *upper1, const F *lower1,
      const F *series2, [[maybe_unused]] size_t length2,
      const F *upper2, const F *lower2,
      utils::CFun<F> auto cfun,
      size_t v,
      size_t w,
      F cutoff
    ) {
      F lb1{utils::INITLB<F>}; // Init with a small negative value: handle numerical instability
      size_t nbands = std::min(length1/2, v);
      // --- --- --- Do L & R Bands
      // First alignment
      lb1 += cfun(series1[0], series2[0]);
      // Manage the case of series of length 1
      if (length1==1) { return (lb1>cutoff) ? utils::PINF<F> : lb1; }
      const size_t last = length1 - 1;
      // Last alignment
      lb1 += cfun(series1[last], series2[last]);
      // L & R bands
      for (size_t i = 1; i<nbands&&lb1<=cutoff; ++i) {
        const auto fixR = last - i;
        F minL = cfun(series1[i], series2[i]);
        F minR = cfun(series1[fixR], series2[fixR]);
        for (size_t j = utils::cap_start_index_to_window(i, w); j<i; ++j) {
          const auto movR = last - j;
          minL = utils::min(minL, cfun(series1[i], series2[j]), cfun(series1[j], series2[i]));
          minR = utils::min(minR, cfun(series1[fixR], series2[movR]), cfun(series1[movR], series2[fixR]));
        }
        lb1 = lb1 + minL + minR;
      }
      // --- --- ---
      if (lb1>cutoff) { return utils::PINF<F>; }

      // --- --- --- Bridge with LB Keogh, continue while we are <= cutoff
      F lb2{lb1};
      const auto end = length1 - nbands;
      for (size_t i = nbands; i<end&&lb1<=cutoff&&lb2<=cutoff; ++i) {
        // Query - envelope candidate
        {
          F qi{series1[i]};
          if (const auto ui{upper2[i]}; qi>ui) { lb1 += cfun(qi, ui); }
          else if (const auto li{lower2[i]}; qi<li) { lb1 += cfun(qi, li); }
        }
        // Candidate - envelope query
        {
          F ci{series2[i]};
          if (const auto ui{upper1[i]}; ci>ui) { lb2 += cfun(ci, ui); }
          else if (const auto li{lower1[i]}; ci<li) { lb2 += cfun(ci, li); }
        }
      }

      lb1 = std::max<F>(lb1, lb2);
      return (lb1>cutoff) ? utils::PINF<F> : lb1;
    }

  } // End of namespace tempo::univariate

} // End of namespace tempo::distance
