#pragma once

#include "../utils.private.hpp"

#include <deque>

namespace tempo::distance {

  namespace univariate {

    /** LB Keogh - only applicable for same-length series.
     * We differentiate between the 'query' series and the 'candidate' series.
     * The query is the series itself;
     * The candidate is represented by its two envelopes that must have been previously computed.
     * @tparam F            Floating type used for the computation
     * @param query           Query series
     * @param query_lengh     Length of the query
     * @param upper           Upper envelope of the candidate series - of length lq (not checked)
     * @param lower           Lower envelope of the candidate series - of length lq (not checked)
     * @param cfun            Cost Function utils::CFun<F>
     * @param cutoff          Cut-off value (strictly) above which we early abandon ("best so far")
     * @return +INF if early abandoned, or the lower bound value
     */
    template<typename F>
    F lb_Keogh(F const *query, size_t query_lengh, F const *upper, F const *lower, utils::CFun<F> auto cfun, F cutoff) {
      // Init
      F lb{0};
      // Main loop, continue while we are <= ub
      for (size_t i = 0; i<query_lengh&&lb<=cutoff; ++i) {
        F qi{query[i]};
        if (const auto ui{upper[i]}; qi>ui) { lb += cfun(qi, ui); }
        else if (const auto li{lower[i]}; qi<li) { lb += cfun(qi, li); }
      }
      return (lb>cutoff) ? utils::PINF<F> : lb;
    }

    /** LB Keogh 2 ways 'joined' - only applicable for same-length series.
     * LB Keogh is not symmetric: for two series s1 and s2, lb_keogh(s1,s2) != lb_keogh(s2, s1).
     * One result is usually tighter than the other, and it is usual to take the maximum of the two computations.
     * This function does that more efficiently by computing both bounds 'j'ointly, and stopping as soon as possible.
     * @tparam F          Floating type used for the computation
     * @param series1     First series data
     * @param length1     First series length
     * @param upper1      First series upper envelope
     * @param lower1      First series lower envelope
     * @param series2     Second series data
     * @param length2     Second series length
     * @param upper2      Second series upper envelope
     * @param lower2      Second series lower envelope
     * @param cfun        Cost Function utils::CFun<F>
     * @param cutoff      Cut-off value (strictly) above which we early abandon ("best so far")
     * @return +INF if early abandoned, or the lower bound value
     */
    template<typename F>
    F lb_Keogh2j(
      F const *series1, size_t length1,
      F const *upper1, F const *lower1,
      F const *series2, [[maybe_unused]] size_t length2,
      F const *upper2, F const *lower2,
      utils::CFun<F> auto cfun,
      F cutoff
    ) {
      // Init
      F lb1{0};
      F lb2{0};
      // Main loop, continue while we are <= ub
      for (size_t i = 0; i<length1&&lb1<=cutoff&&lb2<=cutoff; ++i) {
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
      F lb = std::max<F>(lb1, lb2);
      return (lb>cutoff) ? utils::PINF<F> : lb;
    }


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Envelope computation
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /** Compute the upper and lower envelopes of a series, suitable for LB_Keogh
    *  Implementation based on Lemire's method
    *    Lemire D (2009) Faster retrieval with a two-pass dynamic-time-warping lower bound.
    *    Pattern Recognition 42:2169–2180. https://doi.org/10.1016/j.patcog.2008.11.030
    * @tparam F     Floating type used for the computation
    * @param series Input series
    * @param length Length of the input series
    * @param upper Output array for the upper envelope - Must be able to store 'length' element
    * @param lower Output array for the lower envelope - Must be able to store 'length' element
    * @param w The window for which the envelope is computed.
    */
    template<typename F>
    void get_keogh_envelopes(F const *series, size_t length, F *upper, F *lower, size_t w) {

      // --- Window size adjustment and early exit
      if (length==0) { return; }
      if (w>=length) { w = length - 1; }
      if (w==0) {
        for (size_t i = 0; i<length; ++i) { upper[i] = lower[i] = series[i]; }
        return;
      }

      // --- Initialize the queues with the first w points
      std::deque<size_t> up{0};   // Contains indexes of decreasing values series[idx] (done with (1)). front is max
      std::deque<size_t> lo{0};   // Contains indexes of increasing values series[idx] (done with (2)). front is min
      for (size_t i{1}; i<w; ++i) {
        const F prev{series[i - 1]};
        const F si{series[i]};
        // remark comparison or strict comparison does not matters, hence the else allow to avoid an extra if
        if (prev<=si) {
          do { up.pop_back(); }
          while (!up.empty()&&series[up.back()]<=si);
        } // (1) Remove while si is larger than up[back]
        else {
          do { lo.pop_back(); }
          while (!lo.empty()&&series[lo.back()]>=si);
        }        // (2) Remove while si is smaller than lo[back]
        up.push_back(i);
        lo.push_back(i);
      }

      // --- Go over the series up to length-(w+1)
      // update queue[i+w+1], then update envelopes[i] with front of the queue
      size_t up_front_idx{up.front()};
      F up_front_val{series[up_front_idx]};
      size_t lo_front_idx{lo.front()};
      F lo_front_val{series[lo_front_idx]};
      for (size_t i{0}; i<length - w; ++i) {
        // Update the queues:
        const size_t idx{i + w};
        const F prev{series[idx - 1]}; // Ok as w > 0
        const F si{series[idx]};
        // 1) Evict item preventing monotonicity.
        // If a queue is empty, the item to add is also the new front
        if (prev<=si) {
          do { up.pop_back(); } while (!up.empty()&&series[up.back()]<=si);
          if (up.empty()) {
            up_front_idx = idx;
            up_front_val = series[up_front_idx];
          }
        } else {
          do { lo.pop_back(); } while (!lo.empty()&&series[lo.back()]>=si);
          if (lo.empty()) {
            lo_front_idx = idx;
            lo_front_val = series[lo_front_idx];
          }
        }
        // 2) Push back index, then update envelopes with front indexes
        up.push_back(idx);
        lo.push_back(idx);
        upper[i] = up_front_val; // max over range
        lower[i] = lo_front_val; // min over range
        // 3) trim the front
        if (up_front_idx + w<=i) {
          up.pop_front();
          up_front_idx = up.front();
          up_front_val = series[up_front_idx];
        }
        if (lo_front_idx + w<=i) {
          lo.pop_front();
          lo_front_idx = lo.front();
          lo_front_val = series[lo_front_idx];
        }
      }

      // --- Finish the last w+1 items: values are present already in up_front_val and lo_front_val
      for (size_t i{length - w}; i<length; ++i) {
        // Update the envelope
        upper[i] = up_front_val;
        lower[i] = lo_front_val;
        // Trim the front
        if (up_front_idx + w<=i) {
          up.pop_front();
          up_front_idx = up.front();
          up_front_val = series[up_front_idx];
        }
        if (lo_front_idx + w<=i) {
          lo.pop_front();
          lo_front_idx = lo.front();
          lo_front_val = series[lo_front_idx];
        }
      }
    }

    /** Compute only the upper envelopes of a series.
     *  Implementation based on Lemire's method
     *    Lemire D (2009) Faster retrieval with a two-pass dynamic-time-warping lower bound.
     *    Pattern Recognition 42:2169–2180. https://doi.org/10.1016/j.patcog.2008.11.030
     * @tparam F     Floating type used for the computation
     * @param series Input series
     * @param length Length of the input series
     * @param upper Output array for the upper envelope - Must be able to store 'length' element
     * @param w The window for which the envelope is computed.
     */
    template<typename F>
    void get_keogh_up_envelope(F const *series, size_t length, F *upper, size_t w) {

      // --- Window size adjustment and early exit
      if (length==0) { return; }
      if (w>=length) { w = length - 1; }
      if (w==0) {
        for (size_t i = 0; i<length; ++i) { upper[i] = series[i]; }
        return;
      }

      // --- Initialize the queues with the first w points
      std::deque<size_t> up{0};
      for (size_t i{1}; i<w; ++i) {
        const F prev{series[i - 1]};
        const F si{series[i]};
        if (prev<=si) { do { up.pop_back(); } while (!up.empty()&&series[up.back()]<=si); }
        up.push_back(i);
      }

      // --- Go over the series up to length-(w+1)
      // update queue[i+w+1], then update envelopes[i] with front of the queue
      size_t up_front_idx{up.front()};
      F up_front_val{series[up_front_idx]};
      for (size_t i{0}; i<length - w; ++i) {
        // Update the queues:
        const size_t idx{i + w};
        const F prev{series[idx - 1]}; // Ok as w > 0
        const F si{series[idx]};
        // 1) Evict item preventing monotonicity.
        // If a queue is empty, the item to add is also the new front
        if (prev<=si) {
          do { up.pop_back(); } while (!up.empty()&&series[up.back()]<=si);
          if (up.empty()) {
            up_front_idx = idx;
            up_front_val = series[up_front_idx];
          }
        }
        // 2) Push back index, then update envelopes with front indexes
        up.push_back(idx);
        upper[i] = up_front_val;
        // 3) trim the front
        if (up_front_idx + w<=i) {
          up.pop_front();
          up_front_idx = up.front();
          up_front_val = series[up_front_idx];
        }
      }

      // --- Finish the last w+1 items: values are present already in up_front_val and lo_front_val
      for (size_t i{length - w}; i<length; ++i) {
        // Update the envelope
        upper[i] = up_front_val;
        // Trim the front
        if (up_front_idx + w<=i) {
          up.pop_front();
          up_front_idx = up.front();
          up_front_val = series[up_front_idx];
        }
      }
    }

    /** Compute lower envelopes of a series.
     *  Implementation based on Lemire's method
     *    Lemire D (2009) Faster retrieval with a two-pass dynamic-time-warping lower bound.
     *    Pattern Recognition 42:2169–2180. https://doi.org/10.1016/j.patcog.2008.11.030
     * @tparam F     Floating type used for the computation
     * @param series Input series
     * @param length Length of the input series
     * @param lower Output array for the lower envelope - Must be able to store 'length' element
     * @param w The window for which the envelope is computed.
     */
    template<typename F>
    void get_keogh_lo_envelope(F const *series, size_t length, F *lower, size_t w) {

      // --- Window size adjustment and early exit
      if (length==0) { return; }
      if (w>=length) { w = length - 1; }
      if (w==0) {
        for (size_t i = 0; i<length; ++i) { lower[i] = series[i]; }
        return;
      }

      // --- Initialize the queues with the first w points
      std::deque<size_t> lo{0};   // Contains indexes of increasing values series[idx] (done with (2)). front is min
      for (size_t i{1}; i<w; ++i) {
        const F prev{series[i - 1]};
        const F si{series[i]};
        // remark comparison or strict comparison does not matters, hence the else allow to avoid an extra if
        if (prev>=si) { do { lo.pop_back(); } while (!lo.empty()&&series[lo.back()]>=si); }
        lo.push_back(i);
      }

      // --- Go over the series up to length-(w+1)
      // update queue[i+w+1], then update envelopes[i] with front of the queue
      size_t lo_front_idx{lo.front()};
      F lo_front_val{series[lo_front_idx]};
      for (size_t i{0}; i<length - w; ++i) {
        // Update the queues:
        const size_t idx{i + w};
        const F prev{series[idx - 1]}; // Ok as w > 0
        const F si{series[idx]};
        // 1) Evict item preventing monotonicity.
        // If a queue is empty, the item to add is also the new front
        if (prev>=si) {
          do { lo.pop_back(); } while (!lo.empty()&&series[lo.back()]>=si);
          if (lo.empty()) {
            lo_front_idx = idx;
            lo_front_val = series[lo_front_idx];
          }
        }
        // 2) Push back index, then update envelopes with front indexes
        lo.push_back(idx);
        lower[i] = lo_front_val; // min over range
        // 3) trim the front
        if (lo_front_idx + w<=i) {
          lo.pop_front();
          lo_front_idx = lo.front();
          lo_front_val = series[lo_front_idx];
        }
      }

      // --- Finish the last w+1 items: values are present already in up_front_val and lo_front_val
      for (size_t i{length - w}; i<length; ++i) {
        // Update the envelope
        lower[i] = lo_front_val;
        // Trim the front
        if (lo_front_idx + w<=i) {
          lo.pop_front();
          lo_front_idx = lo.front();
          lo_front_val = series[lo_front_idx];
        }
      }
    }

    namespace {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Vector helper
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

      /** Compute the upper and lower envelopes of a series, suitable for LB_Keogh.
       *  Wrapper for get_envelopes with vector
       * @tparam F      Floating type used for the computation
       * @param series  Constant input series
       * @param upper   Output series - reallocation may occur!
       * @param lower   Output series - reallocation may occur!
       * @param w       The window for which the envelope is computed.
       */
      template<typename F>
      inline void get_keogh_envelopes(std::vector<F> const& series,
                                      std::vector<F>& upper,
                                      std::vector<F>& lower,
                                      size_t w) {
        upper.resize(series.size());
        lower.resize(series.size());
        get_keogh_envelopes(series.data(), series.size(), upper.data(), lower.data(), w);
      }

      /** Compute only the upper envelopes of a series.
       *  Wrapper for get_envelopes with vector
       * @tparam F      Floating type used for the computation
       * @param series  Constant input series
       * @param upper   Output series - reallocation may occur!
       * @param w       The window for which the envelope is computed.
       */
      template<typename F>
      inline void get_keogh_up_envelope(std::vector<F> const& series, std::vector<F>& upper, size_t w) {
        upper.resize(series.size());
        get_keogh_up_envelope(series.data(), series.size(), upper.data(), w);
      }

      /** Compute the lower envelopes of a series.
       *  Wrapper for get_envelopes with vector
       * @tparam F      Floating type used for the computation
       * @param series  Constant input series
       * @param lower   Output series - reallocation may occur!
       * @param w       The window for which the envelope is computed.
       */
      template<typename F>
      inline void get_keogh_lo_envelope(std::vector<F> const& series, std::vector<F>& lower, size_t w) {
        lower.resize(series.size());
        get_keogh_lo_envelope(series.data(), series.size(), lower.data(), w);
      }

    }

  } // End of namespace univariate

} // End of namespace tempo::distance
