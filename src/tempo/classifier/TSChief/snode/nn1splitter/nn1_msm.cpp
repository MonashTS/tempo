#include "nn1_msm.hpp"

#include <tempo/distance/tseries.univariate.hpp>

namespace tempo::classifier::TSChief::snode::nn1splitter {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // MSM Wrapper
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  MSM::MSM(std::string tname, F cost) : BaseDist(std::move(tname)), cost(cost) {}

  F MSM::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::univariate::msm(t1, t2, cost, bsf);
  }

  std::string MSM::get_distance_name() {
    return "MSM:" + std::to_string(cost);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // MSM splitter Generator
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  MSMGen::MSMGen(TransformGetter get_transform, T_GetterState<F> get_cost) :
    get_transform(std::move(get_transform)), get_cost(std::move(get_cost)) {}

  std::unique_ptr<i_Dist> MSMGen::generate(TreeState& state, TreeData const& /* d */, const ByClassMap& /* bcm */) {
    const std::string tn = get_transform(state);
    const F cost = get_cost(state);
    return std::make_unique<MSM>(tn, cost);
  }

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
