#pragma once

#include "../../../utils.hpp"

namespace tempo::univariate {

    /** Lb Keogh, early stopping if above ub
     * @param query The time series being queried
     * @param length_query It's length
     * @param upper Upper envelope from the query from the database
     * @param lower Lower envelope from the query from the database
     * @param ub Current upper bound (best so far)
     * @return A lower bound on the DTW alignment
     */
    [[nodiscard]] double lb_Keogh(
            const double *query, size_t length_query,
            const double *upper, const double *lower,
            double ub
    ) {
        double lb{0};

        for (size_t i = 0; i < length_query && lb < ub; i++) {
            double qi{query[i]};
            if (const auto ui{upper[i]}; qi > ui) {
                lb += square_dist(qi, ui);
            } else if (const auto li{lower[i]}; qi < li) {
                lb += square_dist(qi, li);
            }
        }

        return lb;
    }


    [[nodiscard]] inline double lb_Keogh(
            const std::vector<double> &query,
            const std::vector<double> &upper,
            const std::vector<double> &lower,
            double ub
    ) {
        // lb_Keogh requires same size series
        assert(query.size() == upper.size());
        assert(query.size() == lower.size());
        return lb_Keogh(query.data(), query.size(), upper.data(), lower.data(), ub);
    }

}