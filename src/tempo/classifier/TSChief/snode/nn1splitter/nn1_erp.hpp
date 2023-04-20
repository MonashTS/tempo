#pragma once

#include "nn1dist_base.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  struct ERP : public BaseDist {
    F cfe;
    F gv;
    size_t w;

    ERP(std::string tname, F cfe, F gv, size_t w);

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;

    std::string get_distance_name() override;
  };

  struct ERPGen : public i_GenDist {
    TransformGetter get_transform;
    ExponentGetter get_cfe;
    StatGetter get_gv;
    WindowGetter get_win;

    ERPGen(TransformGetter get_transform, ExponentGetter get_cfe, StatGetter get_gv, WindowGetter get_win);

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& data, const ByClassMap& bcm) override;
  };

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
