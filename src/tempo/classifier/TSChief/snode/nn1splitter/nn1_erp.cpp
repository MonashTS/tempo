#include "nn1_erp.hpp"

#include <tempo/distance/tseries.univariate.hpp>

namespace tempo::classifier::TSChief::snode::nn1splitter {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // ERP Wrapper
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  ERP::ERP(std::string tname, F cfe, F gv, size_t w) : BaseDist(std::move(tname)), cfe(cfe), gv(gv), w(w) {}

  F ERP::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::univariate::erp(t1, t2, cfe, gv, w, bsf);
  }

  std::string ERP::get_distance_name() {
    return "ERP:" + std::to_string(cfe) + ":" + std::to_string(gv) + ":" + std::to_string(w);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // ERP splitter Generator
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  ERPGen::ERPGen(TransformGetter get_transform, ExponentGetter get_cfe, StatGetter get_gv, WindowGetter get_win) :
    get_transform(std::move(get_transform)),
    get_cfe(std::move(get_cfe)),
    get_gv(std::move(get_gv)),
    get_win(std::move(get_win)) {}

  std::unique_ptr<i_Dist> ERPGen::generate(TreeState& state, TreeData const& data, const ByClassMap& bcm) {
    const std::string tn = get_transform(state);
    const F e = get_cfe(state);
    const F gv = get_gv(state, data, bcm, tn);
    const size_t w = get_win(state, data);
    return std::make_unique<ERP>(tn, e, gv, w);
  }

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
