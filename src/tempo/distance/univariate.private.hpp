#pragma once

#include "utils.hpp"
#include "cost_functions.hpp"
// --- --- --- Elastic distances --- --- ---
#include "core/elastic/adtw.hpp"
#include "core/elastic/dtw.hpp"
#include "core/elastic/wdtw.hpp"
#include "core/elastic/erp.hpp"
#include "core/elastic/lcss.hpp"
#include "core/elastic/msm.hpp"
#include "core/elastic/twe.hpp"
// --- --- --- DTW Lower Bound --- --- ---
#include "core/elastic/dtw_lb_keogh.hpp"
#include "core/elastic/dtw_lb_enhanced.hpp"
#include "core/elastic/dtw_lb_webb.hpp"
// --- ------ Lock step distances --- --- ---
#include "core/lockstep/direct.hpp"

#include <cstddef>
#include <vector>

namespace tempo::distance::univariate {

  namespace td = tempo::distance;

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Elastic distances
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  template<typename F>
  F adtw(
    F const *dat1, size_t len1,
    F const *dat2, size_t len2,
    F cfe,
    F penalty,
    F cutoff
  ) {
    if (cfe==1.0) {
      return td::adtw<F>(len1, len2, idx_ad1<F, F const *>(dat1, dat2), penalty, cutoff);
    } else if (cfe==2.0) {
      return td::adtw<F>(len1, len2, idx_ad2<F, F const *>(dat1, dat2), penalty, cutoff);
    } else {
      return td::adtw<F>(len1, len2, idx_ade<F, F const *>(cfe)(dat1, dat2), penalty, cutoff);
    }
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  template<typename F>
  F dtw(
    F const *const dat1, size_t len1,
    F const *const dat2, size_t len2,
    F cfe,
    size_t w,
    F cutoff
  ) {
    if (cfe==1.0) {
      return td::dtw<F>(len1, len2, idx_ad1<F, F const *>(dat1, dat2), w, cutoff);
    } else if (cfe==2.0) {
      return td::dtw<F>(len1, len2, idx_ad2<F, F const *>(dat1, dat2), w, cutoff);
    } else {
      return td::dtw<F>(len1, len2, idx_ade<F, F const *>(cfe)(dat1, dat2), w, cutoff);
    }
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  template<typename F>
  F wdtw(F const *dat1, size_t len1,
         F const *dat2, size_t len2,
         F cfe,
         F const *weights,
         F cutoff
  ) {
    if (cfe==1.0) {
      return td::wdtw<F>(len1, len2, idx_ad1<F, F const *>(dat1, dat2), weights, cutoff);
    } else if (cfe==2.0) {
      return td::wdtw<F>(len1, len2, idx_ad2<F, F const *>(dat1, dat2), weights, cutoff);
    } else {
      return td::wdtw<F>(len1, len2, idx_ade<F, F const *>(cfe)(dat1, dat2), weights, cutoff);
    }
  }

  template<typename F>
  void wdtw_weights(F g, F *weights_array, size_t length, F wmax) {
    td::wdtw_weights(g, weights_array, length, wmax);
  }

  template<typename F>
  std::vector<F> wdtw_weights(F g, size_t length, F wmax) {
    return td::wdtw_weights(g, length, wmax);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  template<typename F>
  F erp(
    F const *const dat1, size_t len1,
    F const *const dat2, size_t len2,
    F cfe,
    F gv,
    size_t w,
    F cutoff
  ) {
    if (cfe==1.0) {
      constexpr auto gv1 = idx_gvad1<F, F const *>;
      return td::erp<F>(len1, len2, gv1(dat1, gv), gv1(dat2, gv), idx_ad1<F, F const *>(dat1, dat2), w, cutoff);
    } else if (cfe==2.0) {
      constexpr auto gv2 = idx_gvad2<F, F const *>;
      return td::erp<F>(len1, len2, gv2(dat1, gv), gv2(dat2, gv), idx_ad2<F, F const *>(dat1, dat2), w, cutoff);
    } else {
      auto gve = idx_gvade<F, F const *>(cfe);
      return td::erp<F>(len1, len2, gve(dat1, gv), gve(dat2, gv), idx_ade<F, F const *>(cfe)(dat1, dat2), w, cutoff);
    }
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  template<typename F>
  F lcss(
    F const *const dat1, size_t len1,
    F const *const dat2, size_t len2,
    F e,
    size_t w,
    F cutoff
  ) {
    return td::lcss<F>(len1, len2, idx_simdiff<F, F const *>(e)(dat1, dat2), w, cutoff);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  template<typename F>
  F msm(F const *data1, size_t length1,
        F const *data2, size_t length2,
        F cost,
        F cutoff
  ) {
    constexpr auto cfli = idx_msm_lines<F, F const *>;
    constexpr auto cfco = idx_msm_cols<F, F const *>;
    constexpr auto cfdi = idx_msm_diag<F, F const *>;
    return td::msm<F>(length1, length2, cfli(data1, data2, cost), cfco(data1, data2, cost), cfdi(data1, data2), cutoff);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  template<typename F>
  F twe(F const *data1, size_t length1,
        F const *data2, size_t length2,
        F nu,
        F lambda,
        F cutoff
  ) {
    constexpr auto cfwarp = univariate::idx_twe_warp<F, F const *>;
    constexpr auto cfmatch = univariate::idx_twe_match<F, F const *>;
    return td::twe<F>(
      length1, length2, cfwarp(data1, nu, lambda), cfwarp(data2, nu, lambda), cfmatch(data1, data2, nu), cutoff
                     );
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // DTW Lower Bound
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  template<typename F>
  F lb_Keogh(F const *query, size_t query_length, F const *upper, F const *lower, F cfe, F cutoff) {
    if (cfe==1.0) {
      constexpr utils::CFun<F> auto cf = ad1<F>;
      return lb_Keogh(query, query_length, upper, lower, cf, cutoff);
    } else if (cfe==2.0) {
      constexpr utils::CFun<F> auto cf = ad2<F>;
      return lb_Keogh(query, query_length, upper, lower, cf, cutoff);
    } else {
      utils::CFun<F> auto cf = ade<F>(cfe);
      return lb_Keogh(query, query_length, upper, lower, cf, cutoff);
    }
  }

  template<typename F>
  F lb_Keogh2j(
    F const *series1, size_t length1, F const *upper1, F const *lower1,
    F const *series2, size_t length2, F const *upper2, F const *lower2,
    F cfe, F cutoff
  ) {
    if (cfe==1.0) {
      constexpr utils::CFun<F> auto cf = ad1<F>;
      return lb_Keogh2j(series1, length1, upper1, lower1, series2, length2, upper2, lower2, cf, cutoff);
    } else if (cfe==2.0) {
      constexpr utils::CFun<F> auto cf = ad2<F>;
      return lb_Keogh2j(series1, length1, upper1, lower1, series2, length2, upper2, lower2, cf, cutoff);
    } else {
      utils::CFun<F> auto cf = ade<F>(cfe);
      return lb_Keogh2j(series1, length1, upper1, lower1, series2, length2, upper2, lower2, cf, cutoff);
    }
  }

  // Warning: the following functions are directly read from
  // #include "core/elastic/dtw_lb_keogh.hpp"
  // template<typename F>
  // void get_keogh_envelopes(F const *series, size_t length, F *upper, F *lower, size_t w);
  //
  // template<typename F>
  // void get_keogh_up_envelope(F const *series, size_t length, F *upper, size_t w);
  //
  // template<typename F>
  // void get_keogh_lo_envelope(F const *series, size_t length, F *lower, size_t w)

  template<typename F>
  F lb_Enhanced(const F *query, size_t query_len,
                const F *candidate, size_t candidate_len,
                const F *candidate_up, const F *candidate_lo,
                F cfe, size_t v, size_t w, F cutoff
  ) {
    if (cfe==1.0) {
      constexpr utils::CFun<F> auto cf = ad1<F>;
      return lb_Enhanced(query, query_len, candidate, candidate_len, candidate_up, candidate_lo, cf, v, w, cutoff);
    } else if (cfe==2.0) {
      constexpr utils::CFun<F> auto cf = ad2<F>;
      return lb_Enhanced(query, query_len, candidate, candidate_len, candidate_up, candidate_lo, cf, v, w, cutoff);
    } else {
      utils::CFun<F> auto cf = ade<F>(cfe);
      return lb_Enhanced(query, query_len, candidate, candidate_len, candidate_up, candidate_lo, cf, v, w, cutoff);
    }
  }

  template<typename F>
  F lb_Enhanced2j(
    const F *series1, size_t length1, const F *upper1, const F *lower1,
    const F *series2, size_t length2, const F *upper2, const F *lower2,
    F cfe, size_t v, size_t w, F cutoff
  ) {
    if (cfe==1.0) {
      constexpr utils::CFun<F> auto cf = ad1<F>;
      return lb_Enhanced2j(series1, length1, upper1, lower1, series2, length2, upper2, lower2, cf, v, w, cutoff);
    } else if (cfe==2.0) {
      constexpr utils::CFun<F> auto cf = ad2<F>;
      return lb_Enhanced2j(series1, length1, upper1, lower1, series2, length2, upper2, lower2, cf, v, w, cutoff);
    } else {
      utils::CFun<F> auto cf = ade<F>(cfe);
      return lb_Enhanced2j(series1, length1, upper1, lower1, series2, length2, upper2, lower2, cf, v, w, cutoff);
    }
  }

  template<typename F>
  F lb_Webb(
    // Series A
    F const *a, size_t a_len,
    F const *a_up, F const *a_lo,
    F const *a_lo_up, F const *a_up_lo,
    // Series B
    F const *b, size_t b_len,
    F const *b_up, F const *b_lo,
    F const *b_lo_up, F const *b_up_lo,
    // Cost function
    F cfe,
    // Others
    size_t w, F cutoff
  ) {
    if (cfe==1.0) {
      constexpr utils::CFun<F> auto cf = ad1<F>;
      return lb_Webb(a, a_len, a_up, a_lo, a_lo_up, a_up_lo, b, b_len, b_up, b_lo, b_lo_up, b_up_lo, cf, w, cutoff);
    } else if (cfe==2.0) {
      constexpr utils::CFun<F> auto cf = ad2<F>;
      return lb_Webb(a, a_len, a_up, a_lo, a_lo_up, a_up_lo, b, b_len, b_up, b_lo, b_lo_up, b_up_lo, cf, w, cutoff);
    } else {
      utils::CFun<F> auto cf = ade<F>(cfe);
      return lb_Webb(a, a_len, a_up, a_lo, a_lo_up, a_up_lo, b, b_len, b_up, b_lo, b_lo_up, b_up_lo, cf, w, cutoff);
    }
  }

  // Warning: the following function are directly read from
  // #include "core/elastic/dtw_lb_webb.hpp"
  // template<typename F>
  // void get_keogh_envelopes_Webb(F const *series, size_t len, F *up, F *lo, F *lower_upper, F *upper_lower, size_t w)


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Lockstep distances
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  /// Direct alignment with cost function exponent, and early abandoning cutoff.
  template<typename F>
  F directa(F const *dat1, size_t len1, F const *dat2, size_t len2, F cfe, F cutoff) {
    if (cfe==1.0) {
      return td::directa<F>(len1, len2, idx_ad1<F, F const *>(dat1, dat2), cutoff);
    } else if (cfe==2.0) {
      return td::directa<F>(len1, len2, idx_ad2<F, F const *>(dat1, dat2), cutoff);
    } else {
      return td::directa<F>(len1, len2, idx_ade<F, F const *>(cfe)(dat1, dat2), cutoff);
    }
  }

} // End of namespace tempo::distance::univariate
