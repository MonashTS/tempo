#pragma once

#include <vector>
#include <tempo/utils/utils.hpp>
#include <tempo/univariate/distances/distances.hpp>

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference implementation
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace reference {

    /// Implementing cost from the original paper
    /// c is the minimal cost of an operation
    inline double get_cost(double xi, double xi1, double yi, double c) {
        if ((xi1 <= xi && xi <= yi) || (xi1 >= xi && xi >= yi)) {
            return c;
        } else {
            return c + std::min(std::fabs(xi - xi1), std::fabs(xi - yi));
        }
    }

    double msm_matrix(
            const double *series1, ssize_t length1,
            const double *series2, ssize_t length2,
            double c);

    /// Wrapper for vector
    inline double msm_matrix(
            const std::vector<double> &series1,
            const std::vector<double> &series2,
            double c) {
        return msm_matrix(series1.data(), series1.size(), series2.data(), series2.size(), c);
    }

} // End of namespace reference
