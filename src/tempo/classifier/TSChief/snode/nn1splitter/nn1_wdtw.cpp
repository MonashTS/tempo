#include "nn1_wdtw.hpp"

#include <tempo/distance/tseries.univariate.hpp>

namespace tempo::classifier::TSChief::snode::nn1splitter {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // WDTW Wrapper
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  WDTW::WDTW(std::string tname, F cfe, F g, std::vector<F>&& weights) :
    BaseDist(std::move(tname)), cfe(cfe), g(g), weights(std::move(weights)) {}

  F WDTW::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::univariate::wdtw(t1, t2, cfe, weights.data(), bsf);
  }

  std::string WDTW::get_distance_name() {
    return "WDTW:" + std::to_string(cfe) + ":" + std::to_string(g);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // WDTW splitter Generator
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  WDTWGen::WDTWGen(TransformGetter gt, ExponentGetter get_cfe, size_t maxl) :
    get_transform(std::move(gt)), get_cfe(std::move(get_cfe)), maxl(maxl) {}

  std::unique_ptr<i_Dist> WDTWGen::generate(TreeState& state, TreeData const& /*data*/, const ByClassMap& /*bcm*/) {
    const std::string tn = get_transform(state);
    const F cfe = get_cfe(state);
    const F g = std::uniform_real_distribution<F>(0, 1)(state.prng);
    return std::make_unique<WDTW>(tn, cfe, g, distance::univariate::wdtw_weights(g, maxl));
  }

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
