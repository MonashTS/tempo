#pragma once

#include <tempo/univariate/distances/dtw/dtw.hpp>
#include <tempo/univariate/distances/dtw/cdtw.hpp>
#include <tempo/univariate/distances/dtw/wdtw.hpp>

#include <tempo/univariate/distances/dtw/lowerbound/envelopes.hpp>
#include <tempo/univariate/distances/dtw/lowerbound/lb_enhanced.hpp>
#include <tempo/univariate/distances/dtw/lowerbound/lb_keogh.hpp>
#include <tempo/univariate/distances/dtw/lowerbound/lb_webb.hpp>

namespace cpp = tempo::univariate;

#include "../../utils.hpp"

namespace pytempo::univariate {

  // --- --- --- --- --- ---
  // DTW
  // --- --- --- --- --- ---

  inline double dtw(nparray series1, nparray series2) {
    check(series1, series2);
    return cpp::dtw(series1.data(), series1.size(), series2.data(), series2.size());
  }

  inline double dtw_ea(nparray series1, nparray series2, double cutoff) {
    check(series1, series2);
    return cpp::dtw(series1.data(), series1.size(), series2.data(), series2.size(), cutoff);
  }



  // --- --- --- --- --- ---
  // CDTW
  // --- --- --- --- --- ---

  inline double cdtw(nparray series1, nparray series2, size_t w) {
    check(series1, series2);
    return cpp::cdtw(series1.data(), series1.size(), series2.data(), series2.size(), w);
  }

  inline double cdtw_ea(nparray series1, nparray series2, size_t w, double cutoff) {
    check(series1, series2);
    return cpp::cdtw(series1.data(), series1.size(), series2.data(), series2.size(), w, cutoff);
  }



  // --- --- --- --- --- ---
  // WDTW
  // --- --- --- --- --- ---

  inline double wdtw(nparray series1, nparray series2, nparray weights) {
    check(series1, series2);
    if (weights.size()<std::max(series1.size(), series2.size())) {
      throw std::invalid_argument("Weights array is too short: must be at least as long as the longest series.");
    }
    return cpp::wdtw(series1.data(), series1.size(), series2.data(), series2.size(), weights.data());
  }

  inline double wdtw_ea(nparray series1, nparray series2, nparray weights, double cutoff) {
    check(series1, series2);
    if (weights.size()<std::max(series1.size(), series2.size())) {
      throw std::invalid_argument("Weights array is too short: must be at least as long as the longest series.");
    }
    return cpp::wdtw(series1.data(), series1.size(), series2.data(), series2.size(), weights.data(), cutoff);
  }

  py::array_t<double> wdtw_weights(size_t length, double g) {
    auto result = py::array_t<double>(length);
    cpp::populate_weights(g, result.mutable_data(), result.size());
    return result;
  }


  // --- --- --- --- --- ---
  // Envelope computation
  // --- --- --- --- --- ---

  using env_t = py::array_t<double>;

  inline std::tuple<env_t, env_t> get_envelopes(nparray series, size_t w) {
    check_dimension(series, 1);
    const auto length = series.size();
    auto up = env_t(length);
    auto lo = env_t(length);
    cpp::get_keogh_envelopes(series.data(), length, up.mutable_data(), lo.mutable_data(), w);
    return {std::move(up), std::move(lo)};
  }

  inline env_t get_envelope_upper(nparray series, size_t w) {
    check_dimension(series, 1);
    const auto length = series.size();
    auto up = env_t(length);
    cpp::get_keogh_up_envelope(series.data(), length, up.mutable_data(), w);
    return up;
  }

  inline env_t get_envelope_lower(nparray series, size_t w) {
    check_dimension(series, 1);
    const auto length = series.size();
    auto lo = env_t(length);
    cpp::get_keogh_lo_envelope(series.data(), length, lo.mutable_data(), w);
    return lo;
  }


  // --- --- --- --- --- ---
  // Lower bound computation
  // --- --- --- --- --- ---

  inline double lb_keogh(nparray query, nparray candidate_up_env, nparray candidate_lo_env, double cutoff) {
    check_dimension(query, 1);
    check_same_length(query, candidate_up_env);
    check_same_length(query, candidate_lo_env);
    return cpp::lb_Keogh(query.data(), query.size(), candidate_up_env.data(), candidate_lo_env.data(), cutoff);
  }

  inline double lb_keogh2(
    nparray query, nparray query_up_env, nparray query_lo_env,
    nparray candidate, nparray candidate_up_env, nparray candidate_lo_env, double cutoff) {
    check(query, candidate);
    check_same_length(query, candidate);
    double res1 = cpp::lb_Keogh(query.data(), query.size(), candidate_up_env.data(), candidate_lo_env.data(), cutoff);
    if (res1<=cutoff) {
      double res2 = cpp::lb_Keogh(candidate.data(), candidate.size(), query_up_env.data(), query_lo_env.data(), cutoff);
      return std::max<double>(res1, res2);
    } else {
      return res1;
    }
  }

  inline double lb_keogh2j(
    nparray query, nparray query_up_env, nparray query_lo_env,
    nparray candidate, nparray candidate_up_env, nparray candidate_lo_env, double cutoff) {
    check(query, candidate);
    check_same_length(query, candidate);
    return cpp::lb_Keogh2j(
      query.data(), query.size(), query_up_env.data(), query_lo_env.data(),
      candidate.data(), candidate.size(), candidate_up_env.data(), candidate_lo_env.data(),
      cutoff);
  }

  inline double
  lb_enhanced(nparray query, nparray candidate, nparray candidate_up_env, nparray candidate_lo_env, size_t w,
    double cutoff, size_t v) {
    check(query, candidate);
    check_same_length(query, candidate);
    return cpp::lb_Enhanced(
      query.data(), query.size(),
      candidate.data(), candidate.size(), candidate_up_env.data(), candidate_lo_env.data(),
      v, w, cutoff);
  }

  inline double
  lb_enhanced2j(
    nparray query, nparray query_up_env, nparray query_lo_env,
    nparray candidate, nparray candidate_up_env, nparray candidate_lo_env, size_t w,
    double cutoff, size_t v) {
    check(query, candidate);
    check_same_length(query, candidate);
    return cpp::lb_Enhanced2j(
      query.data(), query.size(), query_up_env.data(), query_lo_env.data(),
      candidate.data(), candidate.size(), candidate_up_env.data(), candidate_lo_env.data(),
      v, w, cutoff);
  }

  inline double
  lb_webb(
    nparray query, nparray q_up, nparray q_lo, nparray q_lo_up, nparray q_up_lo,
    nparray candidate, nparray c_up, nparray c_lo, nparray c_lo_up, nparray c_up_lo,
    size_t w, double cutoff) {
    check(query, candidate);
    check_same_length(query, candidate);
    return cpp::lb_Webb(
      query.data(), query.size(), q_up.data(), q_lo.data(), q_lo_up.data(), q_up_lo.data(),
      candidate.data(), candidate.size(), c_up.data(), c_lo.data(), c_lo_up.data(), c_up_lo.data(),
      w, cutoff);
  }


    // --- --- --- --- --- ---
    // Python module
    // --- --- --- --- --- ---

    inline void init_dtw(py::module& m) {

      // --- --- --- DTW

      m.def("dtw", &dtw,
        "DTW between two series.",
        "serie1"_a, "serie2"_a
      );

      m.def("dtw", &dtw_ea,
        "DTW between two series. With pruning & early abandoning cut-off.",
        "serie1"_a, "serie2"_a, "cutoff"_a
      );

      // --- --- --- CDTW

      m.def("cdtw", &cdtw,
        "Constrained DTW between two series. \"w\" is the half-window size.",
        "serie1"_a, "serie2"_a, "w"_a
      );

      m.def("cdtw", &cdtw_ea,
        "Constrained DTW between two series. \"w\" is the half-window size. With pruning & early abandoning cut-off.",
        "serie1"_a, "serie2"_a, "w"_a, "cutoff"_a
      );

      // --- --- --- WDTW

      m.def("wdtw", &wdtw,
        "Weighted DTW between two series, using an array of weights.",
        "serie1"_a, "serie2"_a, "weights"_a
      );

      m.def("wdtw", &wdtw_ea,
        "Weighted DTW between two series, using an array of weights. With pruning & early abandoning cut-off.",
        "serie1"_a, "serie2"_a, "weights"_a, "cutoff"_a
      );

      m.def("wdtw_weights", &wdtw_weights,
        "Generate a numpy array of weights suitable for WDTW.",
        "length"_a, "g"_a);

      // --- --- --- Envelopes

      m.def("get_envelopes", &get_envelopes,
        "Compute the (upper, lower) envelopes",
        "series"_a, "w"_a);

      m.def("get_envelope_upper", &get_envelope_upper,
        "Compute the upper envelope",
        "series"_a, "w"_a);

      m.def("get_envelope_lower", &get_envelope_lower,
        "Compute the lower envelope",
        "series"_a, "w"_a);

      // --- --- --- Lower bound

      m.def("lb_keogh", &lb_keogh,
        "Compute LB-Keogh given a series ('query') and the envelopes of the other series ('candidate')",
        "query"_a, "up_envelope"_a, "lo_envelope"_a, "cutoff"_a = tempo::POSITIVE_INFINITY<double>);

      m.def("lb_keogh2", &lb_keogh2,
        "Cascade two computation of LB-Keogh: first with the query and the envelopes of the candidate, "
        "then, if the result is below the cutoff, with the candidate and the envelopes of the query. "
        " If cascading occurred, returns the maximum value."
        "query"_a, "query_up_envelope"_a, "query_lo_envelope"_a,
        "candidate"_a, "candidate_up_envelope"_a, "candidate_lo_envelope"_a,
        "cutoff"_a = tempo::POSITIVE_INFINITY<double>);

      m.def("lb_keogh2j", &lb_keogh2j,
        "Joined LB-Keogh2: perform two LB-Keoghs (query and env candidate + candidate and env query) at the same time",
        "query"_a, "query_up_envelope"_a, "query_lo_envelope"_a,
        "candidate"_a, "candidate_up_envelope"_a, "candidate_lo_envelope"_a,
        "cutoff"_a = tempo::POSITIVE_INFINITY<double>);

      m.def("lb_enhanced", &lb_enhanced,
        "LB-Enhanced - you must also provide the window. By default, computes 5 bands on each side ('v' argument).",
        "query"_a, "candidate"_a, "candidate_up_env"_a, "candidate_lo_env"_a, "w"_a,
        "cutoff"_a = tempo::POSITIVE_INFINITY<double>, "v"_a = 5);

      m.def("lb_enhanced2j", &lb_enhanced2j,
        "LB-Enhanced 2 Joined - you must also provide the window. By default, computes 5 bands on each side ('v' argument). "
        "This version 'bridges the two sides' with LB-Keogh2j instead of LB-Keogh. "
        "Recommended.",
        "query"_a, "query_up_env"_a, "query_lo_env"_a,
        "candidate"_a, "candidate_up_env"_a, "candidate_lo_env"_a, "w"_a,
        "cutoff"_a = tempo::POSITIVE_INFINITY<double>, "v"_a = 5);

      m.def("lb_webb", &lb_webb,
        "LB-Webb - requires the window and four envelopes per series",
        "query"_a, "q_up"_a, "q_lo"_a, "q_lo_up"_a, "q_up_lo"_a,
        "candidate"_a, "c_up"_a, "c_lo"_a, "c_lo_up"_a, "c_up_lo"_a,
        "w"_a, "cutoff"_a = tempo::POSITIVE_INFINITY<double>);
    }

  }