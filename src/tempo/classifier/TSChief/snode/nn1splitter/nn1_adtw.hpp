#pragma once

#include "nn1dist_base.hpp"

namespace tempo::classifier::TSChief::snode::nn1splitter {

  /** BaseDist ADTW wrapper
   *  Wrap a parameterized call to the ADTW distance
   */
  struct ADTW : public BaseDist {
    F cfe;
    F penalty;

    ADTW(std::string tname, F cfe, F penalty);

    F eval(const TSeries& t1, const TSeries& t2, F bsf) override;

    std::string get_distance_name() override;
  };


  /** ADTW Splitter Generator
   *  Generate a randomly parameterized ADTW VCFE splitter.
   *  Must be initialized with pre-sampled max penalties per (cost function exponent, transform).
   *  use the provided 'do_sampling' function for that!
   */
  struct ADTWGen : public i_GenDist {

    static constexpr F omega_exponent = 5.0;

    TransformGetter get_transform;
    ExponentGetter get_fce;
    std::map<std::tuple<F, std::string>, std::vector<F>> penalties;

    ADTWGen(TransformGetter gt, ExponentGetter get_cfe, std::map<std::tuple<F, std::string>, std::vector<F>> penalties);

    std::unique_ptr<i_Dist> generate(TreeState& state, TreeData const& /*data*/ , const ByClassMap& /*bcm*/) override;

    /**
     * Create a vector of penalties per cost function exponent and transform.
     * Note: we directly precompute all the penalties, not just the max penalties (hence the map of vectors).
     * @param exponent          list of possible cost function exponent
     * @param transforms        list of transforms
     * @param train_data        training data - only the transforms from the above list woll be considered
     * @param SAMPLE_SIZE       how many sample to take
     * @param prng              source of randomness for sampling
     * @param NUMBER_PENALTY    how many penalties (parameters) to generate per (cost function exponent, transform)
     * @param OMEGA_EXPONENT    omega used when generating penalties
     * Given a sampled max penalty S, penalties are generated with
     *  S*(i/NUMBER_PENALTY)^OMEGA_EXPONENT for i in [0, NUMBER_PENALTY[
     * @return a mapping (cost function exponent, transform)-> vec<penalties>
     */
    static std::map<std::tuple<F, std::string>, std::vector<F>> do_sampling(
      std::vector<F> const& exponent,
      std::vector<std::string> const& transforms,
      std::map<std::string, DTS> const& train_data,
      size_t SAMPLE_SIZE,
      PRNG& prng,
      size_t NUMBER_PENALTY = 100,
      F OMEGA_EXPONENT = 5.0
    );

  };

} // End of namespace tempo::classifier::PF2::snode::nn1splitter