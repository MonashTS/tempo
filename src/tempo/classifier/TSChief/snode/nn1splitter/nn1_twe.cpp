#include "nn1_twe.hpp"

#include <tempo/distance/tseries.univariate.hpp>

namespace tempo::classifier::TSChief::snode::nn1splitter {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // TWE Wrapper
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  TWE::TWE(std::string tname, F nu, F lambda) : BaseDist(std::move(tname)), nu(nu), lambda(lambda) {}

  F TWE::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::univariate::twe(t1, t2, nu, lambda, bsf);
  }

  std::string TWE::get_distance_name() {
    return "TWE:" + std::to_string(nu) + ":" + std::to_string(lambda);
  }


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // TWE splitter Generator
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  TWEGen::TWEGen(TransformGetter get_transform, T_GetterState<F> get_nu, T_GetterState<F> get_lambda) :
    get_transform(std::move(get_transform)), get_nu(std::move(get_nu)), get_lambda(std::move(get_lambda)) {}

  std::unique_ptr<i_Dist> TWEGen::generate(TreeState& state, TreeData const& /* d */, const ByClassMap& /* bcm */) {
    const std::string tn = get_transform(state);
    const F nu = get_nu(state);
    const F lambda = get_lambda(state);
    return std::make_unique<TWE>(tn, nu, lambda);
  }

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
