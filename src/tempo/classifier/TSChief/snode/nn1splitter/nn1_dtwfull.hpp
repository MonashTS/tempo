#pragma once

#include "nn1dist_base.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  struct DTWFull : public BaseDist {
    F cfe;

    DTWFull(std::string tname, F cfe);

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;

    std::string get_distance_name() override;
  };

  struct DTWFullGen : public i_GenDist {
    TransformGetter get_transform;
    ExponentGetter get_fce;

    DTWFullGen(TransformGetter gt, ExponentGetter get_cfe);

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& /*data*/, const ByClassMap& /* bcm */) override;
  };

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
