#pragma once

#include "nn1dist_base.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  struct MSM : public BaseDist {
    F cost;

    MSM(std::string tname, F cost);

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;

    std::string get_distance_name() override;
  };

  struct MSMGen : public i_GenDist {
    TransformGetter get_transform;
    T_GetterState<F> get_cost;

    MSMGen(TransformGetter get_transform, T_GetterState<F> get_cost);

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& /* d */, const ByClassMap& /* bcm */) override;
  };

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
