#ifndef TEST_REFERENCES_EAPDISTANCES_DTW_HPP
#define TEST_REFERENCES_EAPDISTANCES_DTW_HPP

#include <vector>
#include <tempo/utils/utils.hpp>
#include <tempo/univariate/elastic_distances/distances.hpp>

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference implementation, with square euclidean distance
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace reference {

    double dtw_matrix(
            const double *series1, std::size_t length1,
            const double *series2, std::size_t length2);

    /// Wrapper for vector
    inline double dtw_matrix(
            const std::vector<double> &series1,
            const std::vector<double> &series2) {
        return dtw_matrix(series1.data(), series1.size(), series2.data(), series2.size());
    }

} // End of namespace references

#endif
