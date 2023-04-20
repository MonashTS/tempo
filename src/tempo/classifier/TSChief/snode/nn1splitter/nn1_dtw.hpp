#pragma once

#include "nn1dist_base.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  struct DTW : public BaseDist {
    F cfe;
    size_t w;

    DTW(std::string tname, F cfe, size_t w);

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;

    std::string get_distance_name() override;
  };

  struct DTWGen : public i_GenDist {
    TransformGetter get_transform;
    ExponentGetter get_cfe;
    WindowGetter get_win;

    DTWGen(TransformGetter gt, ExponentGetter get_cfe, WindowGetter get_win);

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& data, const ByClassMap& /* bcm */) override;
  };

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
