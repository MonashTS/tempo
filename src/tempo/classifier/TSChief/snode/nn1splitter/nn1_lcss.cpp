#include "nn1_lcss.hpp"

#include <tempo/distance/tseries.univariate.hpp>

namespace tempo::classifier::TSChief::snode::nn1splitter {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // LCSS Wrapper
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  LCSS::LCSS(std::string tname, F epsilon, size_t w) : BaseDist(std::move(tname)), epsilon(epsilon), w(w) {}

  F LCSS::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::univariate::lcss(t1, t2, epsilon, w, bsf);
  }

  std::string LCSS::get_distance_name() { return "LCSS:" + std::to_string(epsilon) + ":" + std::to_string(w); }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // LCSS splitter Generator
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  LCSSGen::LCSSGen(TransformGetter get_transform, StatGetter get_epsilon, WindowGetter get_win) :
    get_transform(std::move(get_transform)), get_epsilon(std::move(get_epsilon)), get_win(std::move(get_win)) {}

  std::unique_ptr<i_Dist> LCSSGen::generate(TreeState& state, TreeData const& data, const ByClassMap& bcm) {
    const std::string tn = get_transform(state);
    const F epsilon = get_epsilon(state, data, bcm, tn);
    const size_t w = get_win(state, data);
    return std::make_unique<LCSS>(tn, epsilon, w);
  }

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
