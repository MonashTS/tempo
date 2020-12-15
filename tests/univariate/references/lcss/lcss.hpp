#pragma once

#include <vector>
#include <tempo/utils/utils.hpp>
#include <tempo/univariate/elastic_distances/distances.hpp>

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference implementation, with square euclidean distance
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace reference {

    /// Naive LCSS implementation. Reference code.
    double lcss_matrix(
            const double *series1, ssize_t length1,
            const double *series2, ssize_t length2,
            double epsilon,
            size_t w);

    /// Wrapper for vector
    inline double lcss_matrix(
            const std::vector<double> &series1,
            const std::vector<double> &series2,
            double epsilon,
            size_t w) {
        return lcss_matrix(series1.data(), series1.size(), series2.data(), series2.size(), epsilon, w);
    }

} // End of namespace reference
