#pragma once

#include <vector>
#include <tempo/utils/utils.hpp>
#include <tempo/univariate/distances/distances.hpp>

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference implementation, with square euclidean distance
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace reference {

    double cdtw_matrix(
            const double *series1, ssize_t length1,
            const double *series2, ssize_t length2,
            size_t w);

    /// Wrapper for vector
    inline double cdtw_matrix(
            const std::vector<double> &series1,
            const std::vector<double> &series2,
            size_t w) {
        return cdtw_matrix(series1.data(), series1.size(), series2.data(), series2.size(), w);
    }

} // End of namespace references
