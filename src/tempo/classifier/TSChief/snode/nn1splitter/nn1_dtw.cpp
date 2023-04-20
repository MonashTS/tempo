#include "nn1_dtw.hpp"
#include <tempo/distance/tseries.univariate.hpp>

namespace tempo::classifier::TSChief::snode::nn1splitter {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // DTW Wrapper
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  DTW::DTW(std::string tname, F cfe, size_t w) : BaseDist(std::move(tname)), cfe(cfe), w(w) {}

  F DTW::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::univariate::dtw(t1, t2, cfe, w, bsf);
  }

  std::string DTW::get_distance_name() {
    return "DTW:" + std::to_string(cfe) + ":" + std::to_string(w);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // DTW splitter Generator
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  DTWGen::DTWGen(TransformGetter gt, ExponentGetter get_cfe, WindowGetter get_win) :
    get_transform(std::move(gt)), get_cfe(std::move(get_cfe)), get_win(std::move(get_win)) {}

  std::unique_ptr<i_Dist> DTWGen::generate(TreeState& state, TreeData const& data, const ByClassMap& /* bcm */) {
    const std::string tn = get_transform(state);
    const F e = get_cfe(state);
    const size_t w = get_win(state, data);
    return std::make_unique<DTW>(tn, e, w);
  }

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
