#pragma once

#include <vector>
#include "utils.hpp"

namespace tempo::distance::univariate {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Elastic
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// ADTW with cost function exponent, penalty, and EAP cutoff.
  template<typename F>
  F adtw(
    F const *data1, size_t length1,
    F const *data2, size_t length2,
    F cfe,
    F penalty,
    F cutoff
  );

  /// DTW with cost function exponent, warping window length, and EAP cutoff.
  /// Use window=NO_WINDOW to use unconstrained DTW.
  template<typename F>
  F dtw(
    F const *data1, size_t length1,
    F const *data2, size_t length2,
    F cfe,
    size_t window,
    F cutoff
  );

  /// WDTW with cost function exponent, weights and EAP cutoff.
  template<typename F>
  F wdtw(F const *data1, size_t length1,
         F const *data2, size_t length2,
         F cfe,
         F const *weights,
         F cutoff
  );

  /// Populate a pointed array of size length by weights suitable for WDTW.
  template<typename F>
  void wdtw_weights(F g, F *weights_array, size_t length, F wmax = 1);

  /// Helper for the above, generating a vector on the fly
  template<typename F>
  std::vector<F> wdtw_weights(F g, size_t length, F wmax = 1);

  /// ERP with cost function exponent, gap value, warping window, and EAP cutoff.
  /// Use window=NO_WINDOW to use unconstrained ERP.
  template<typename F>
  F erp(F const *data1, size_t length1,
        F const *data2, size_t length2,
        F cfe,
        F gap_value,
        size_t window,
        F cutoff
  );

  /// LCSS with epsilon, warping window, and EAP cutoff.
  /// Use window=NO_WINDOW to use unconstrained LCSS.
  template<typename F>
  F lcss(F const *data1, size_t length1,
         F const *data2, size_t length2,
         F epsilon,
         size_t window,
         F cutoff
  );

  /// MSM with cost and EAP cutoff.
  template<typename F>
  F msm(F const *data1, size_t length1,
        F const *data2, size_t length2,
        F cost,
        F cutoff
  );

  /// TWE with stiffness (nu) and penalty (lambda) parameters, and EAP cutoff.
  template<typename F>
  F twe(F const *data1, size_t length1,
        F const *data2, size_t length2,
        F nu,
        F lambda,
        F cutoff
  );

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // DTW Lower Bounds
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// LB Keogh for a query and a candidate represented by its envelopes.
  /// Only use for same length series. Tunable cost function exponent similar to dtw.
  template<typename F>
  F lb_Keogh(F const *query, size_t query_length, F const *upper, F const *lower, F cfe, F cutoff);

  /// LB Keogh 2 ways done 'jointly' - for same length series,
  /// with tunable cost function exponent similar to dtw
  template<typename F>
  F lb_Keogh2j(
    F const *series1, size_t length1, F const *upper1, F const *lower1,
    F const *series2, size_t length2, F const *upper2, F const *lower2,
    F cfe, F cutoff
  );

  /// Given a series, compute both the upper and lower envelopes for a window w
  /// Write the results in upper and lower (must point to buffers of size >= length)
  template<typename F>
  void get_keogh_envelopes(F const *series, size_t length, F *upper, F *lower, size_t w);

  /// Given a series, compute the upper envelope for a window w
  /// Write the results in upper (must point to a buffer of size >= length)
  template<typename F>
  void get_keogh_up_envelope(F const *series, size_t length, F *upper, size_t w);

  /// Given a series, compute the lower envelope for a window w
  /// Write the results in lower (must point to a buffer of size >= length)
  template<typename F>
  void get_keogh_lo_envelope(F const *series, size_t length, F *lower, size_t w);

  /// LB Enhanced for a query and a candidate series with its envelopes.
  /// Only use for same length series. Tunable cost function exponent similar to dtw.
  /// 'v' is the number of LR bands, speed/tightness trade-off (faster = 0, tighter = length/2)
  template<typename F>
  F lb_Enhanced(const F *query, size_t query_length,
                const F *candidate, size_t candidate_length, const F *candidate_upper, const F *candidate_lower,
                F cfe, size_t v, size_t w, F cutoff);

  /// LB Enhanced 2 ways done 'jointly' for two series and their envelopes?
  /// Only use for same length series. Tunable cost function exponent similar to dtw.
  /// 'v' is the number of LR bands, speed/tightness trade-off (faster = 0, tighter = length/2)
  template<typename F>
  F lb_Enhanced2j(
    const F *series1, size_t length1, const F *upper1, const F *lower1,
    const F *series2, size_t length2, const F *upper2, const F *lower2,
    F cfe, size_t v, size_t w, F cutoff
  );

  /// LB Webb for two same length series sa and sb, with their envelopes,
  /// and the lower envelope of their upper envelopes,
  /// and the upper envelope of their lower envelopes.
  /// Tunable cost function exponent similar to dtw.
  template<typename F>
  F lb_Webb(
    // Series A
    F const *sa, size_t length_sa,
    F const *upper_sa, F const *lower_sa,
    F const *lower_upper_sa, F const *upper_lower_sa,
    // Series B
    F const *sb, size_t length_sb,
    F const *upper_sb, F const *lower_sb,
    F const *lower_upper_sb, F const *upper_lower_sb,
    // Others
    F cfe, size_t w, F cutoff
  );

  /// Given a series, compute all the envelopes require for lb_Webb.
  /// Write the results in upper, lower, lower_upper and upper_lower, which must point to large enough buffers.
  template<typename F>
  void get_keogh_envelopes_Webb(
    F const *series, size_t length, F *upper, F *lower, F *lower_upper, F *upper_lower, size_t w
  ) {
    get_keogh_envelopes(series, length, upper, lower, w);
    get_keogh_lo_envelope(upper, length, lower_upper, w);
    get_keogh_up_envelope(lower, length, upper_lower, w);
  }


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Lockstep
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Direct alignment with cost function exponent, and early abandoning cutoff.
  template<typename F>
  F directa(F const *data1, size_t length1, F const *data2, size_t length2, F cfe, F cutoff);

} // End of namespace tempo::distance::univariate
