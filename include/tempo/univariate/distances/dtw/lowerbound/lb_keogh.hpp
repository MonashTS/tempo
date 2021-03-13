#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/univariate/distances/distances.hpp>
#include <tempo/tseries/transform.hpp>

namespace tempo::univariate {

  // Todo - separate in equal length and non equal length.
  /*
  template<typename FloatType, auto dist = square_dist<FloatType>>
  [[nodiscard]] FloatType lb_Keogh(
    const FloatType* query, size_t lq,
    const FloatType* upper, const FloatType* lower, size_t lc,
    size_t w,
    FloatType ub
  ) {
    // Pre check - Empty series
    if (lq==0 && lc==0) { return {FloatType(0.0)}; }
    else if ((lq==0)!=(lc==0)) { return POSITIVE_INFINITY<FloatType>; }

    // Pre check - Diff length
    if (lq==lc) {
      // Init
      FloatType lb{0};
      // Main loop, continue while we are <= ub
      for (size_t i = 0; i<lq && lb<=ub; ++i) {
        FloatType qi{query[i]};
        if (const auto ui{upper[i]}; qi>ui) { lb += dist(qi, ui); }
        else if (const auto li{lower[i]}; qi<li) { lb += dist(qi, li); }
      }
      return (lb>ub) ? POSITIVE_INFINITY<FloatType> : lb;
    } else {
      // Pre check - does the window allow an alignment
      const auto lmax = std::max<size_t>(lq, lc);
      const auto lmin = std::min<size_t>(lq, lc);
      if (w>lmax) { w = lmax; }
      if (lmax-lmin>w) { return POSITIVE_INFINITY<FloatType>; }
      // Init
      FloatType lb{0};
      // Main loop, continue while we are <= ub
      for (size_t i = 0; i<lmin && lb<=ub; ++i) {
        FloatType qi{query[i]};
        if (const auto ui{upper[i]}; qi>ui) { lb += dist(qi, ui); }
        else if (const auto li{lower[i]}; qi<li) { lb += dist(qi, li); }
      }
      // Secondary loop, only if the smallest series is the candidate
      if (lmin==lc) {
        const auto ui = upper[lmin-1];
        const auto li = lower[lmin-1];
        for (size_t i = lmin; i<lmax && lb<=ub; ++i) {
          FloatType qi{query[i]};
          if (qi>ui) { lb += dist(qi, ui); }
          else if (qi<li) { lb += dist(qi, li); }
        }
      }
      return (lb>ub) ? POSITIVE_INFINITY<FloatType> : lb;
    }
  }
   */

  /** LB Keogh. Only applicable for same-size series!
   * @param query Query series
   * @param lq Length of the query
   * @param upper Upper envelope of the candidate
   * @param lower Lower envelope of the candidate
   * @param cutoff Cut-off value above which we early abandon (best so fat)
   * @return +Inf if early abandoned, or the lower bound value
   */
  template<typename FloatType, auto dist = square_dist<FloatType>>
  [[nodiscard]] FloatType lb_Keogh(
    const FloatType* query, size_t lq,
    const FloatType* upper, const FloatType* lower,
    FloatType cutoff
  ) {
    // Init
    FloatType lb{0};
    // Main loop, continue while we are <= ub
    for (size_t i = 0; i<lq && lb<=cutoff; ++i) {
      FloatType qi{query[i]};
      if (const auto ui{upper[i]}; qi>ui) { lb += dist(qi, ui); }
      else if (const auto li{lower[i]}; qi<li) { lb += dist(qi, li); }
    }
    return (lb>cutoff) ? POSITIVE_INFINITY<FloatType> : lb;
  }

  template<typename FloatType, auto dist = square_dist<FloatType>>
  [[nodiscard]] inline FloatType lb_Keogh(
    const std::vector<FloatType>& query,
    const std::vector<FloatType>& upper,
    const std::vector<FloatType>& lower,
    FloatType ub
  ) {
    // lb_Keogh requires same size series
    assert(query.size()==upper.size());
    assert(query.size()==lower.size());
    return lb_Keogh<FloatType, dist>(query.data(), query.size(), upper.data(), lower.data(), ub);
  }

  template<typename FloatType, auto dist = square_dist<FloatType>>
  [[nodiscard]] FloatType lb_Keogh2j(
    const FloatType* query, size_t lq,
    const FloatType* qu, const FloatType* ql,
    const FloatType* candidate, [[maybe_unused]] size_t lc,
    const FloatType* cu, const FloatType* cl,
    FloatType ub
  ) {
    // Init
    FloatType lb1{0};
    FloatType lb2{0};
    // Main loop, continue while we are <= ub
    for (size_t i = 0; i<lq && lb1<=ub && lb2<=ub; ++i) {
      // Query - envelope candidate
      {
        FloatType qi{query[i]};
        if (const auto ui{cu[i]}; qi>ui) { lb1 += dist(qi, ui); }
        else if (const auto li{cl[i]}; qi<li) { lb1 += dist(qi, li); }
      }
      // Candidate - envelope query
      {
        FloatType ci{candidate[i]};
        if (const auto ui{qu[i]}; ci>ui) { lb2 += dist(ci, ui); }
        else if (const auto li{ql[i]}; ci<li) { lb2 += dist(ci, li); }
      }
    }
    FloatType lb = std::max<FloatType>(lb1, lb2);
    return (lb>ub) ? POSITIVE_INFINITY<FloatType> : lb;
  }


  template<typename FloatType, typename LabelType>
  struct KeoghEnvelopesTransformer {
    using TS = TSeries<FloatType, LabelType>;
    using SRC = TransformHandle<std::vector<TS>, FloatType, LabelType>;

    struct Env {
      std::vector<FloatType> lo;
      std::vector<FloatType> up;
    };

    /// Create the transform. Do not add to the src's dataset.
    [[nodiscard]] Transform transform(const SRC& src, size_t w) {
      if ((src.dataset->get_header()).get_ndim()!=1) { throw std::invalid_argument("Dataset is not univariate"); }
      // --- Transform identification
      auto name = "keogh_envelopes("+std::to_string(w)+")";
      auto parent = src.get_transform().get_name_components();
      // --- Compute data
      const std::vector<TS>& src_vec = src.get();
      std::vector<Env> output;
      output.reserve(src_vec.size());
      for (const auto& ts: src_vec) {
        const size_t l = ts.length();
        Env env;
        env.lo.resize(l);
        env.up.resize(l);
        get_keogh_envelopes<FloatType>(ts.data(), l, env.up.data(), env.lo.data(), w);
        output.template emplace_back(std::move(env));
      }
      // --- Create transform
      Capsule capsule = tempo::make_capsule<std::vector<Env>>(std::move(output));
      const void* ptr = tempo::capsule_ptr<std::vector<Env>>(capsule);
      return Transform(std::move(name), std::move(parent), std::move(capsule), ptr);
    }

    /// Create the transform and add it to the src's dataset. Return the corresponding handle.
    [[nodiscard]] TransformHandle<std::vector<Env>, FloatType, LabelType> transform_and_add(SRC& src, size_t w){
      auto tr = transform(src, w);
      return src.dataset->template add_transform<std::vector<Env>>(std::move(tr));
    }


  };

} // End of namespace tempo::univariate