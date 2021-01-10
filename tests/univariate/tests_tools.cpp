#include "tests_tools.hpp"

/** Testing tools */
namespace ttools {

    /** Random number generators for constant parameters */
    unsigned int base_seed{72267943};
    PRNG prng{base_seed};

    /** Windows ratio */
    const vector<double> wratios{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};

    /** ERP parameters */
    const vector<double> gvalues = {0, 0.01, 0.5, 1, 10, 100};

    /** LCS parameters */
    const vector<double> epsilons = tempo::rand::generate_random_real_vector(prng, 15, 0.0, 0.1);

    /** MSM parameters */
    const vector<double> msm_costs = {0, 0.01, 0.5, 1, 10, 100};

    /** TWE parameters */
    const vector<double> twe_nus = tempo::rand::generate_random_real_vector(prng, 7, 0.0, maxv);
    const vector<double> twe_lambdas = tempo::rand::generate_random_real_vector(prng, 7, 0.0, maxv);

    /** WDTW parameters */
    // Factor for the weights generation
    const vector<double> weight_factors = tempo::rand::generate_random_real_vector(prng, 5, 0.0, 1.0);

} // End of namespace testing tools
