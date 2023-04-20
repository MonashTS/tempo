#include "nn1_adtw.hpp"

#include <tempo/distance/tseries.univariate.hpp>

namespace tempo::classifier::TSChief::snode::nn1splitter {

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // ADTW Wrapper
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  ADTW::ADTW(std::string tname, tempo::F cfe, tempo::F penalty)
    : BaseDist(std::move(tname)), cfe(cfe), penalty(penalty) {}

  F ADTW::eval(const TSeries& t1, const TSeries& t2, F bsf) {
    return distance::univariate::adtw(t1, t2, cfe, penalty, bsf);
  }

  std::string ADTW::get_distance_name() { return "ADTW:" + std::to_string(cfe) + ":" + std::to_string(penalty); }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // ADTW Splitter Generator
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  ADTWGen::ADTWGen(TransformGetter gt, ExponentGetter get_cfe,
                   std::map<std::tuple<F, std::string>, std::vector<F>> penalties) :
    get_transform(std::move(gt)),
    get_fce(std::move(get_cfe)),
    penalties(std::move(penalties)) {}

  std::unique_ptr<i_Dist> ADTWGen::generate(TreeState& state, TreeData const& /*data*/ , const ByClassMap& /*bcm*/) {
    const std::string tn = get_transform(state);
    const F e = get_fce(state);
    F penalty = utils::pick_one(penalties.at({e, tn}), state.prng);
    return std::make_unique<ADTW>(tn, e, penalty);
  }

  std::map<std::tuple<F, std::string>, std::vector<F>> ADTWGen::do_sampling(
    std::vector<F> const& exponent,
    std::vector<std::string> const& transforms,
    std::map<std::string, DTS> const& train_data,
    size_t SAMPLE_SIZE,
    PRNG& prng,
    size_t NUMBER_PENALTY,
    F OMEGA_EXPONENT
  ) {
    const F NBP = (F)NUMBER_PENALTY;

    // Function's result accumulator
    std::map<std::tuple<F, std::string>, std::vector<F>> result;

    // For each transform
    for (auto const& tn : transforms) {
      auto const& dts = train_data.at(tn);
      if (dts.size()<=1) { throw std::invalid_argument("DataSplit transform " + tn + " as less than 2 values"); }

      // For each cost function exponent
      for (auto const& e : exponent) {

        // --- Sample
        tempo::utils::StddevWelford welford;
        std::uniform_int_distribution<> distrib(0, (int)dts.size() - 1);
        for (size_t i = 0; i<SAMPLE_SIZE; ++i) {
          const auto& q = dts[distrib(prng)];
          const auto& s = dts[distrib(prng)];
          const F cost = distance::univariate::directa(q, s, e, utils::PINF);
          welford.update(cost);
        }
        F max_penalties = welford.get_mean();

        // --- Compute and store penalties
        std::vector<F> penalties;
        penalties.reserve(NUMBER_PENALTY);
        penalties.push_back(0.0);
        for (size_t i = 1; i<NUMBER_PENALTY; ++i) {
          const F penalty = std::pow((F)(F)i/NBP, OMEGA_EXPONENT)*max_penalties;
          penalties.push_back(penalty);
        }

        // --- Store mapping
        result[std::tuple(e, tn)] = std::move(penalties);
      }
    }

    return result;
  }

} // End of namespace tempo::classifier::PF2::snode::nn1splitter
