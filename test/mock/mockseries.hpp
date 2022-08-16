#pragma once

#include <cstddef>
#include <optional>
#include <random>
#include <vector>
#include <algorithm>

namespace mock {

  /// Mocker class - init with a
  template<typename F=double, typename PRNG=std::mt19937_64>
  struct Mocker {

    // --- --- --- Fields, open for configuration

    // Random number generator - should be init in the constructor
    unsigned int _seed;
    PRNG _prng;
    // Dimension of the series
    size_t _dim{1};
    // Length of the series
    size_t _minl{20};   // variable, min
    size_t _maxl{30};   // variable, max
    size_t _fixl{25};   // fixed
    // Possible values of the series
    F _minv{-10};
    F _maxv{10};

    // Parameters
    std::vector<double> adtw_penalties;
    std::vector<double> wratios{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    std::vector<double> gvalues{0.0, 0.01, 0.5, 1, 10, 100};
    std::vector<double> msm_costs = {0, 0.01, 0.5, 1, 10, 100};
    std::vector<double> twe_nus = {0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1};
    std::vector<double> twe_lambdas = {0, 0.011111111, 0.022222222, 0.033333333, 0.044444444,
                                       0.055555556, 0.066666667,
                                       0.077777778, 0.088888889, 0.1};
    std::vector<double> epsilons = {0.1, 0.2, 0.3, 0.8, 0.9, 1, 2, 5};

    // --- --- --- Constructor

    /** Build a mocker with a random seed. If none is given, one is generated */
    explicit Mocker(std::optional<unsigned int> seed = {}) {
      static_assert(std::is_floating_point_v<F>);

      // Create a PRNG
      if (seed.has_value()) { _seed = seed.value(); }
      else { _seed = std::random_device()(); }
      _prng = PRNG(_seed);

      // Generate penalties for ADTW
      double step = std::pow(_maxv - _minv, 2)/100;
      adtw_penalties.push_back(0);
      for (size_t i = 1; i<=100; ++i) { adtw_penalties.push_back(i*step); }
    }


    // --- --- --- Methods

    /** Random size between min and max */
    [[nodiscard]] inline size_t get_size(size_t minl, size_t maxl) {
      auto dist = std::uniform_int_distribution<std::size_t>(minl, maxl);
      return dist(_prng);
    }

    /** Generate a vector of a given size*_dim with random real values in the half-closed interval [minv, maxv[. */
    [[nodiscard]] std::vector<F> randvec(size_t size, F minv, F maxv) {
      std::uniform_real_distribution<F> udist{minv, maxv};
      auto generator = [this, &udist]() { return udist(_prng); };
      std::vector<F> v(size*_dim);
      std::generate(v.begin(), v.end(), generator);
      return v;
    }

    /** Generate a vector of _fixl size with random real values in the half-closed interval [_minv, _maxv[.*/
    [[nodiscard]] std::vector<F> randvec() { return randvec(_fixl, _minv, _maxv); }

    /** Generate a dataset of fixed length series with nbitems, with values in [_minv, _maxv[ */
    [[nodiscard]] std::vector<std::vector<double>> vec_randvec(size_t nbitems) {
      std::vector<std::vector<double>> set;
      for (size_t i = 0; i<nbitems; ++i) {
        auto series = randvec(_fixl, _minv, _maxv);
        assert(series.data()!=nullptr);
        set.push_back(std::move(series));
      }
      return set;
    }

    /** Generate a vector of a random size between [_minl, _maxl]
     * with random real values in the half-closed interval [_minv, _maxv[.*/
    [[nodiscard]] std::vector<F> rs_randvec() {
      size_t l = get_size(_minl, _maxl);
      return randvec(l, _minv, _maxv);
    }

    /** Generate a dataset of variable length series with nbtimes, with values in [_minv, _maxv[ */
    [[nodiscard]] std::vector<std::vector<double>> vec_rs_randvec(size_t nbitems) {
      std::vector<std::vector<double>> set;
      for (size_t i = 0; i<nbitems; ++i) {
        auto series = rs_randvec();
        assert(series.data()!=nullptr);
        set.push_back(std::move(series));
      }
      return set;
    }

  };

} // End of namespace mockseries
