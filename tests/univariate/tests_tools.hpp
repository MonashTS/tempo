#ifndef EAPDISTANCES_TTOOLS_HPP
#define EAPDISTANCES_TTOOLS_HPP

#include <catch.hpp>
#include <tempo/utils/utils.hpp>

/** Testing tools */
namespace ttools {
    using namespace std;
    using namespace tempo;

    // Default number of series to generate
    constexpr size_t def_nbitems{2000};

    // Default series size, fixed -- range for variable
    constexpr size_t def_fixed_size{15};
    constexpr size_t def_min_size{0};
    constexpr size_t def_max_size{15};

    // Default value range for series
    constexpr double minv{-5};
    constexpr double maxv{5};

    // Pseudo random number generator
    using PRNG = mt19937_64;

    /** Random size between min and max */
    inline size_t get_size(PRNG &prng, size_t min_size = def_min_size, size_t max_size = def_max_size) {
        auto dist = std::uniform_int_distribution<std::size_t>(min_size, max_size);
        return dist(prng);
    }

    /** Fixed length, random vector of double, with values in [minv, maxv[ */
    inline vector<double> get_series_data_fixed_length(PRNG& prng, size_t fixed_size = def_fixed_size){
        return generate_random_real_vector(prng, fixed_size, minv, maxv);
    }

    /** Variable length [min_size, max_size], with values in [minv, maxv[ */
    inline vector<double> get_series_data_variable_length(PRNG &prng, size_t min_size = def_min_size, size_t max_size = def_max_size){
        return generate_random_real_vector(prng, get_size(prng, min_size, max_size), minv, maxv);
    }

    /** Generate a dataset of fixed length series with nbitems, with values in [minv, maxv[ */
    inline vector<vector<double>> get_set_fixed_length(PRNG& prng, int nbitems, size_t fixed_size = def_fixed_size) {
        vector<vector<double>> set;
        for (int i = 0; i < nbitems; ++i) {
            auto series = get_series_data_fixed_length(prng, fixed_size);
            assert(series.data() != nullptr);
            set.push_back(std::move(series));
        }
        return set;
    }

    /** Generate a dataset of variable length series with nbitems, with values in [minv, maxv[ */
    inline vector<vector<double>> get_set_variable_length(PRNG &prng, int nbitems, size_t min_size = def_min_size, size_t max_size = def_max_size){
        vector<vector<double>> set;
        for (int i = 0; i < nbitems; ++i) {
            auto series = get_series_data_variable_length(prng, min_size, max_size);
            set.push_back(std::move(series));
        }
        return set;
    }


    /** Random number generators for constant parameters */
    extern unsigned int base_seed;
    extern PRNG prng;

    /** Windows ratio */
    extern const vector<double> wratios;

    /** ERP parameters */
    extern const vector<double> gvalues;

    /** LCS parameters */
    extern const vector<double> epsilons;

    /** MSM parameters */
    extern const vector<double> msm_costs;

    /** TWE parameters */
    extern const vector<double> twe_nus;
    extern const vector<double> twe_lambdas;

    /** WDTW parameters */
    // Factor for the weights generation
    extern const vector<double> weight_factors;


} // End of namespace testing tools

#endif