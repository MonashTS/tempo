#include "nn1_dtwfull.hpp"

#include <tempo/distance/tseries.univariate.hpp>

namespace tempo::classifier::TSChief::snode::nn1splitter {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // DTWFull Wrapper
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  DTWFull::DTWFull(std::string tname, F cfe) : BaseDist(std::move(tname)), cfe(cfe) {}

  F DTWFull::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::univariate::dtw(t1, t2, cfe, utils::NO_WINDOW, bsf);
  }

  std::string DTWFull::get_distance_name() { return "DTWFull:" + std::to_string(cfe); }


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // DTWFull splitter Generator
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  DTWFullGen::DTWFullGen(TransformGetter gt, ExponentGetter get_cfe) :
    get_transform(std::move(gt)), get_fce(std::move(get_cfe)) {}

  std::unique_ptr<i_Dist> DTWFullGen::generate(TreeState& state, TreeData const& /*data*/, const ByClassMap& /*bcm*/) {
    const std::string tn = get_transform(state);
    const F e = get_fce(state);
    return std::make_unique<DTWFull>(tn, e);
  }

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
