#include "nn1_directa.hpp"

#include <tempo/distance/tseries.univariate.hpp>

namespace tempo::classifier::TSChief::snode::nn1splitter {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // DA Wrapper
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  DA::DA(std::string tname, F cfe) : BaseDist(std::move(tname)), cfe(cfe) {}

  F DA::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::univariate::directa(t1, t2, cfe, bsf);
  }

  std::string DA::get_distance_name() {
    return "DA:" + std::to_string(cfe);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // DA splitter Generator
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  DAGen::DAGen(TransformGetter gt, ExponentGetter ge) :
    get_transform(std::move(gt)), get_cfe(std::move(ge)) {}

  std::unique_ptr<i_Dist> DAGen::generate(TreeState& state, TreeData const& /*data*/, const ByClassMap& /* bcm */) {
    const std::string tn = get_transform(state);
    const F e = get_cfe(state);
    return std::make_unique<DA>(tn, e);
  }

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
