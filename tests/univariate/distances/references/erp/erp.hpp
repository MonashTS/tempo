#pragma once

#include <vector>
#include <tempo/utils/utils.hpp>
#include <tempo/univariate/distances/distances.hpp>

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference implementation, with square euclidean distance
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace reference {

    /// Reference implementation
    double erp_matrix(
            const double *series1, size_t length1_,
            const double *series2, size_t length2_,
            double gValue,
            size_t w
    );

    /// Wrapper for vector
    inline double erp_matrix(
            const std::vector<double> &series1,
            const std::vector<double> &series2,
            double g, size_t w) {
        return erp_matrix(series1.data(), series1.size(), series2.data(), series2.size(), g, w);
    }

} // End of namespace reference

