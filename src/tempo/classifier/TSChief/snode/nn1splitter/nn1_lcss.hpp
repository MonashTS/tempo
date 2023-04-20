#pragma once

#include <tempo/utils/utils.hpp>
#include <tempo/dataset/tseries.hpp>
#include <tempo/distance/tseries.univariate.hpp>

#include "nn1dist_base.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  struct LCSS : public BaseDist {
    F epsilon;
    size_t w;

    LCSS(std::string tname, F epsilon, size_t w);

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;

    std::string get_distance_name() override;
  };

  struct LCSSGen : public i_GenDist {
    TransformGetter get_transform;
    StatGetter get_epsilon;
    WindowGetter get_win;

    LCSSGen(TransformGetter get_transform, StatGetter get_epsilon, WindowGetter get_win);

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& data, const ByClassMap& bcm) override;
  };

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
