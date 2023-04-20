#pragma once

#include "nn1dist_base.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  struct TWE : public BaseDist {
    F nu;
    F lambda;

    TWE(std::string tname, F nu, F lambda);

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;

    std::string get_distance_name() override;
  };

  struct TWEGen : public i_GenDist {
    TransformGetter get_transform;
    T_GetterState<F> get_nu;
    T_GetterState<F> get_lambda;

    TWEGen(TransformGetter get_transform, T_GetterState<F> get_nu, T_GetterState<F> get_lambda);

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& /* d */, const ByClassMap& /* bcm */) override;
  };

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
