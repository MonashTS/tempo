#include "binder_common.hpp"

#include "tempo/distance/utils.hpp"
#include "tempo/distance/univariate.hpp"
namespace tdu = tempo::distance::univariate;
constexpr F PINF = tempo::distance::utils::PINF<F>;
constexpr size_t NO_WINDOW = tempo::distance::utils::NO_WINDOW;


// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Elastic distances
namespace ed {

  F adtw(nparray s1, nparray s2, F cf_exponent, F penalty, F cutoff) {
    check_univariate(s1, s2);
    return tdu::adtw(USE(s1), USE(s2), cf_exponent, penalty, cutoff);
  }

  F dtw(nparray s1, nparray s2, F cf_exponent, std::optional<size_t> op_window, F cutoff) {
    check_univariate(s1, s2);
    if (op_window) {
      return tdu::dtw(USE(s1), USE(s2), cf_exponent, op_window.value(), cutoff);
    } else {
      return tdu::dtw(USE(s1), USE(s2), cf_exponent, NO_WINDOW, cutoff);
    }
  }

  F wdtw(nparray s1, nparray s2, F cf_exponent, nparray weights, F cutoff) {
    return tdu::wdtw(USE(s1), USE(s2), cf_exponent, weights.data(), cutoff);
  }

  nparray_t wdtw_weights(F g, size_t length, F wmax) {
    nparray_t array(length);
    tdu::wdtw_weights(g, array.mutable_data(), length, wmax);
    return std::move(array);
  }

  F erp(nparray s1, nparray s2, F cf_exponent, F gap_value, std::optional<size_t> op_window, F cutoff) {
    check_univariate(s1, s2);
    if (op_window) {
      return tdu::erp(USE(s1), USE(s2), cf_exponent, gap_value, op_window.value(), cutoff);
    } else {
      return tdu::erp(USE(s1), USE(s2), cf_exponent, gap_value, NO_WINDOW, cutoff);
    }
  }

  F lcss(nparray s1, nparray s2, F epsilon, std::optional<size_t> op_window, F cutoff) {
    check_univariate(s1, s2);
    if (op_window) {
      return tdu::lcss(USE(s1), USE(s2), epsilon, op_window.value(), cutoff);
    } else {
      return tdu::lcss(USE(s1), USE(s2), epsilon, NO_WINDOW, cutoff);
    }
  }

  F msm(nparray s1, nparray s2, F cost, F cutoff) {
    return tdu::msm(USE(s1), USE(s2), cost, cutoff);
  }

  F twe(nparray s1, nparray s2, F nu, F lambda, F cutoff) {
    return tdu::twe(USE(s1), USE(s2), nu, lambda, cutoff);
  }

  inline void init(py::module& m) {

    m.def(
      "adtw", &adtw, R"pbdoc(
      Amerced Dynamic Time Warping (ADTW) - EAP implementation.
      Configurable cost function exponent (as 'e' in |s1(i) - s2(j)|^e).
    )pbdoc", "s1"_a, "s2"_a, "cf_exponent"_a, "penalty"_a, "cutoff"_a = PINF
    );

    m.def(
      "dtw", &dtw, R"pbdoc(
        Dynamic Time Warping (DTW) - EAP implementation.
        Configurable cost function exponent (as 'e' in |s1(i)-s2(j)|^e).
        Use window=None for unconstrained DTW, or specify a window length 'w',
        allowing to align points within 'w' timestamps ('w' before and 'w' after the considered point).
    )pbdoc", "s1"_a, "s2"_a, "cf_exponent"_a, "window"_a = std::nullopt, "cutoff"_a = PINF
    );

    m.def(
      "wdtw", &wdtw, R"pbdoc(
        Weighted Dynamic Time Warping (WDTW) - EAP implementation.
        Configurable cost function exponent (as 'e' in |s1(i)-s2(j)|^e).
        Generate the weights with the 'wdtw_weights' function.
    )pbdoc", "s1"_a, "s2"_a, "cf_exponent"_a, "weights"_a, "cutoff"_a = PINF
    );

    m.def(
      "wdtw_weights", &wdtw_weights, R"pbdoc(
        Generate a numpy array of weights for Weighted Dynamic Time Warping (WDTW).
        'g' set the level of penalisation;
        'length' must be the maximum length of the series that could be use with these weights
        'wmax' scaling - default to 1, as recommended in the WDTW paper.
    )pbdoc", "g"_a, "length"_a, "wmax"_a = 1
    );

    m.def(
      "erp", &erp, R"pbdoc(
        Edit Distance with Real Penalty (ERP) - EAP implementation.
        Configurable cost function exponent (as 'e' in |s1(i)-s2(j)|^e).
        Use window=None for unconstrained ERP, or specify a window length 'w',
        allowing to align points within 'w' timestamps ('w' before and 'w' after the considered point).
    )pbdoc", "s1"_a, "s2"_a, "cf_exponent"_a, "gap_value"_a, "window"_a = std::nullopt, "cutoff"_a = PINF
    );

    m.def(
      "lcss", &lcss, R"pbdoc(
        Longest Common SubSequence (LCSS) - EAP implementation.
        Use window=None for unconstrained LCSS, or specify a window length 'w',
        allowing to align points within 'w' timestamps ('w' before and 'w' after the considered point).
    )pbdoc", "s1"_a, "s2"_a, "epsilon"_a, "window"_a = std::nullopt, "cutoff"_a = PINF
    );

    m.def(
      "msm", &msm, R"pbdoc(
        Move Split Merge (MSM) - EAP implementation.
    )pbdoc", "s1"_a, "s2"_a, "cost"_a, "cutoff"_a = PINF
    );

    m.def(
      "twe", &twe, R"pbdoc(
        Time Warp Edit (TWE) - EAP implementation.
        With stiffness parameter nu and penalty lambda.
    )pbdoc", "s1"_a, "s2"_a, "nu"_a, "lambda"_a, "cutoff"_a = PINF
    );

  }

} // End of namespace ed

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// DTW lower bound
namespace lb_dtw {

  F lb_Keogh(nparray query, nparray upper, nparray lower, F cf_exponent, F cutoff) {
    check_univariate(query);
    check_univariate(upper);
    check_univariate(lower);
    check_same_length(query, upper);
    check_same_length(query, lower);
    return tdu::lb_Keogh(USE(query), upper.data(), lower.data(), cf_exponent, cutoff);
  }

  F lb_Keogh2j(
    nparray series1, nparray upper1, nparray lower1, nparray series2, nparray upper2, nparray lower2, F cfe, F cutoff
  ) {
    return tdu::lb_Keogh2j(
      USE(series1), upper1.data(), lower1.data(), USE(series2), upper2.data(), lower2.data(), cfe, cutoff
    );
  }

  std::tuple<nparray_t, nparray_t> get_keogh_envelopes(nparray series, size_t w) {
    check_univariate(series);
    nparray_t upper(series.size());
    nparray_t lower(series.size());
    tdu::get_keogh_envelopes(USE(series), upper.mutable_data(), lower.mutable_data(), w);
    return {std::move(upper), std::move(lower)};
  }

  nparray_t get_keogh_up_envelope(nparray series, size_t w) {
    check_univariate(series);
    nparray_t buf(series.size());
    tdu::get_keogh_up_envelope(USE(series), buf.mutable_data(), w);
    return std::move(buf);
  }

  nparray_t get_keogh_lo_envelope(nparray series, size_t w) {
    check_univariate(series);
    nparray_t buf(series.size());
    tdu::get_keogh_lo_envelope(USE(series), buf.mutable_data(), w);
    return std::move(buf);
  }

  F lb_Enhanced(nparray query, nparray candidate, nparray candidate_upper, nparray candidate_lower, F cf_exponent,
                size_t v, size_t w, F cutoff
  ) {
    check_univariate(query);
    check_univariate(candidate);
    check_univariate(candidate_upper);
    check_univariate(candidate_lower);
    check_same_length(query, candidate);
    check_same_length(query, candidate_upper);
    check_same_length(query, candidate_lower);
    if (v>query.size()/2) { throw std::invalid_argument("The number of bands is too large."); }
    return tdu::lb_Enhanced(
      USE(query), USE(candidate), candidate_upper.data(), candidate_lower.data(), cf_exponent, v, w, cutoff
    );
  }

  F lb_Enhanced2j(
    nparray series1, nparray upper1, nparray lower1,
    nparray series2, nparray upper2, nparray lower2,
    F cf_exponent, size_t v, size_t w, F cutoff
  ) {
    check_univariate(series1);
    check_univariate(upper1);
    check_univariate(lower1);
    check_univariate(series2);
    check_univariate(upper2);
    check_univariate(lower2);
    check_same_length(series1, upper1);
    check_same_length(series1, upper1);
    check_same_length(series1, series2);
    check_same_length(series2, lower2);
    check_same_length(series2, lower2);
    if (v>series1.size()/2) { throw std::invalid_argument("The number of bands is too large."); }
    return tdu::lb_Enhanced2j(USE(series1), upper1.data(), lower1.data(), USE(series2), upper2.data(), lower2.data(),
                              cf_exponent, v, w, cutoff
    );
  }

  F lb_Webb(
    nparray sa, nparray upper_sa, nparray lower_sa, nparray lower_upper_sa, nparray upper_lower_sa,
    nparray sb, nparray upper_sb, nparray lower_sb, nparray lower_upper_sb, nparray upper_lower_sb,
    F cf_exponent, size_t w, F cutoff
  ) {
    check_univariate(sa);
    check_univariate(upper_sa);
    check_univariate(lower_sa);
    check_univariate(lower_upper_sa);
    check_univariate(upper_lower_sa);
    check_univariate(sb);
    check_univariate(upper_sb);
    check_univariate(lower_sb);
    check_univariate(lower_upper_sb);
    check_univariate(upper_lower_sb);
    //
    check_same_length(sa, upper_sa);
    check_same_length(sa, lower_sa);
    check_same_length(sa, lower_upper_sa);
    check_same_length(sa, upper_lower_sa);
    check_same_length(sa, sb);
    check_same_length(sb, upper_sb);
    check_same_length(sb, lower_sb);
    check_same_length(sb, lower_upper_sb);
    check_same_length(sb, upper_lower_sb);
    //
    return tdu::lb_Webb(
      USE(sa), upper_sa.data(), lower_sa.data(), lower_upper_sa.data(), upper_lower_sa.data(),
      USE(sb), upper_sb.data(), lower_sb.data(), lower_upper_sb.data(), upper_lower_sb.data(),
      cf_exponent, w, cutoff
    );
  }

  std::tuple<nparray_t, nparray_t, nparray_t, nparray_t> get_keogh_envelopes_Webb(
    nparray series, size_t w
  ) {
    check_univariate(series);
    const auto len = series.size();
    nparray_t upper(len);
    nparray_t lower(len);
    nparray_t lower_upper(len);
    nparray_t upper_lower(len);
    tdu::get_keogh_envelopes(USE(series), upper.mutable_data(), lower.mutable_data(), w);
    tdu::get_keogh_lo_envelope(USE(upper), lower_upper.mutable_data(), w);
    tdu::get_keogh_up_envelope(USE(lower), upper_lower.mutable_data(), w);
    return {std::move(upper), std::move(lower), std::move(lower_upper), std::move(upper_lower)};
  }

  inline void init(py::module& m) {

    m.def(
      "lb_keogh", &lb_Keogh, R"pbdoc(
        LB Keogh between a query and a candidate represented by its upper and lower envelopes.
        Configurable cost function exponent (as 'e' in |s1(i)-s2(j)|^e).
        Early abandon with +Inf if the result is (strictly) above cutoff.
    )pbdoc", "query"_a, "upper"_a, "lower"_a, "cf_exponent"_a, "cutoff"_a = PINF
    );

    m.def(
      "lb_keogh2j", &lb_Keogh2j, R"pbdoc(
        LB Keogh, 2 ways joined, between two same-length series and their envelopes.
        Usually more efficient than taking the max(lbKeogh(s1, s2), lbKeogh(s2, s1)).
        Configurable cost function exponent (as 'e' in |s1(i)-s2(j)|^e).
        Early abandon with +Inf if the result is (strictly) above cutoff.
    )pbdoc", "s1"_a, "upper1"_a, "lower1"_a, "s2"_a, "upper2"_a, "lower2"_a, "cf_exponent"_a, "cutoff"_a = PINF
    );

    m.def(
      "envelopes", &get_keogh_envelopes, R"pbdoc("
        Compute the upper and lower envelopes of the series for the given window, using Lemire's method.
        Return a tuple (upper, lower).
      )pbdoc", "series"_a, "window"_a
    );

    m.def(
      "upper_envelope", &get_keogh_up_envelope, R"pbdoc("
        Compute the upper envelope of the series for the given window, using Lemire's method.
      )pbdoc", "series"_a, "window"_a
    );

    m.def(
      "lower_envelope", &get_keogh_lo_envelope, R"pbdoc("
        Compute the lower envelope of the series for the given window, using Lemire's method.
      )pbdoc", "series"_a, "window"_a
    );

    m.def(
      "lb_enhanced", &lb_Enhanced, R"pbdoc("
        LB Enhanced between two same-length series and the envelopes of the second one.
        The number of bands is a tradeoff between speed and tightness,
        and must be an integer between 0 (equivalent to lb Keogh) and ceil(length/2)
        Configurable cost function exponent (as 'e' in |s1(i)-s2(j)|^e).
        Early abandon with +Inf if the result is (strictly) above cutoff.
      )pbdoc", "s1"_a, "s2"_a, "s2_upper"_a, "s2_lower"_a, "cf_exponent"_a, "nb_bands"_a, "window"_a, "cutoff"_a = PINF
    );

    m.def(
      "lb_enhanced2j", &lb_Enhanced2j, R"pbdoc("
        LB Enhanced 2 ways joined, between two same-length series and their envelopes.
        The number of bands is a tradeoff between speed and tightness,
        and must be an integer between 0 (equivalent to lb Keogh) and ceil(length/2)
        Configurable cost function exponent (as 'e' in |s1(i)-s2(j)|^e).
        Early abandon with +Inf if the result is (strictly) above cutoff.
      )pbdoc",
      "s1"_a, "s1_upper"_a, "s1_lower"_a,
      "s2"_a, "s2_upper"_a, "s2_lower"_a,
      "cf_exponent"_a, "nb_bands"_a, "window"_a, "cutoff"_a = PINF
    );

    m.def(
      "lb_webb", &lb_Webb, R"pbdoc("
        LB Webb between two same-length series and their envelopes.
        See webb_envelopes.
        Configurable cost function exponent (as 'e' in |s1(i)-s2(j)|^e).
        Early abandon with +Inf if the result is (strictly) above cutoff.
      )pbdoc",
      "s1"_a, "s1_upper"_a, "s1_lower"_a, "s1_lower_upper"_a, "s1_upper_lower"_a,
      "s2"_a, "s2_upper"_a, "s2_lower"_a, "s2_lower_upper"_a, "s2_upper_lower"_a,
      "cf_exponent"_a, "window"_a, "cutoff"_a = PINF
    );

    m.def(
      "webb_envelopes", &get_keogh_envelopes_Webb, R"pbdoc("
        Compute all the envelopes of a series required by LB Webb.
        The 'upper' and 'lower' envelopes produced by this function can be used with other lower bounds.
        Return a tuple of envelopes (upper, lower, lower_upper, upper_lower),
        where 'lower_upper' is the lower envelope of the upper envelope,
        and 'upper_lower' is the upper envelope of the lower envelope.
      )pbdoc", "series"_a, "window"_a
    );

  }

} // End of namespace lb_dtw

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// LockStep distances
namespace lsd {

  F directa(nparray s1, nparray s2, F cf_exponent, F cutoff) {
    check_univariate(s1, s2);
    return tdu::directa(USE(s1), USE(s2), cf_exponent, cutoff);
  }

  inline void init(py::module& m) {

    m.def(
      "directa", &directa, R"pbdoc(
      Direct Alignment - Early Abandoned Implementation.
      Configurable cost function exponent (as 'e' in |s1(i) - s2(j)|^e).
    )pbdoc", "s1"_a, "s2"_a, "cf_exponent"_a, "cutoff"_a = PINF
    );

  }

} // end of namespace lsd


// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace univariate_distance {

  void init(py::module& m) {
    auto mod_distance = m.def_submodule("distance");
    ed::init(mod_distance);
    lb_dtw::init(mod_distance);
    lsd::init(mod_distance);
  }

}
