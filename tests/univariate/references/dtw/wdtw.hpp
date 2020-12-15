#pragma once

#include <vector>
#include <tempo/utils/utils.hpp>
#include <tempo/univariate/elastic_distances/distances.hpp>


// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference implementation, with square euclidean distance
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace reference {

    /** Reimplementation of the "modified logistic weight function (MLWF)" from the original paper
     * "Weighted dynamic time warping for time series classification"
     * @param i Index of the point in [1..m] (m=length of the sequence)
     * @param mc Mid point of the sequence (m/2)
     * @param g "Controls the level of penalization for the points with larger phase difference".
     *        range [0, +inf), usually in [0.01, 0.6].
     *        Some examples:
     *        * 0: constant weight
     *        * 0.05: nearly linear weights
     *        * 0.25: sigmoid weights
     *        * 3: two distinct weights between half sequences
     * @param wmax Upper bound for the weight parameter. Keep it to 1
     * @return
     */
    inline double mlwf(double i, double mc, double g, double wmax=1){
        return wmax/(1+std::exp(-g*(i - mc)));
    }

    /// Reference implementation on a matrix
    double wdtw_matrix(const double *series1, size_t length1_, const double *series2, size_t length2_, const double *weights);

    /// Wrapper for vector
    inline double wdtw_matrix(
            const std::vector<double> &series1,
            const std::vector<double> &series2,
            const std::vector<double> &weights
            ) {
        assert(series1.size() <= weights.size());
        assert(series2.size() <= weights.size());
        return wdtw_matrix(series1.data(), series1.size(), series2.data(), series2.size(), weights.data());
    }

} // End of namespace references
