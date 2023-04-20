#pragma once

#include "nn1dist_base.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  struct WDTW : public BaseDist {
    F cfe;
    F g;
    std::vector<F> weights;

    WDTW(std::string tname, F cfe, F g, std::vector<F>&& weights);

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;

    std::string get_distance_name() override;
  };

  struct WDTWGen : public i_GenDist {
    TransformGetter get_transform;
    ExponentGetter get_cfe;
    size_t maxl;

    WDTWGen(TransformGetter gt, ExponentGetter get_cfe, size_t maxl);

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& /*data*/, const ByClassMap& /*bcm*/) override;
  };

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
