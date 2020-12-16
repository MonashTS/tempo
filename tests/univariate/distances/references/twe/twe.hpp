#pragma once

#include <vector>
#include <tempo/utils/utils.hpp>
#include <tempo/univariate/distances/distances.hpp>

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference implementation, with square euclidean distance
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace reference {

    /** Reference implementation by Pierre-François Marteau
     * @param ta pointer to first series
     * @param la length of first series
     * @param tb pointer to second series
     * @param lb length of second series
     * @param nu Should be >0, "stiffness" of the measure (Minkowski’s Distance has infinite stiffness, DTW has null stiffness)
     * @param lambda Minimal cost of delete operations
     * @return
     */
    [[nodiscard]] double twe_Marteau(const double *ta, ssize_t la, const double *tb, ssize_t lb, double nu, double lambda);

    /// Wrapper for vector
    [[nodiscard]] double inline twe_Marteau(
            const std::vector<double> &series1,
            const std::vector<double> &series2,
            double nu, double lambda
    ) {
        return twe_Marteau(series1.data(), series1.size(), series2.data(), series2.size(), nu, lambda);
    }

} // End of namespace reference

