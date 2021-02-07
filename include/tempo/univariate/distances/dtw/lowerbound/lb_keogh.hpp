#pragma once

#include "../../../../utils/utils.hpp"
#include "../../distances.hpp"

namespace tempo::univariate {

    /** Lb Keogh, early stopping if above ub
     * @param query The time series being queried
     * @param length_query It's length
     * @param upper Upper envelope from the query from the database
     * @param lower Lower envelope from the query from the database
     * @param ub Current upper bound (best so far)
     * @return A lower bound on the DTW alignment
     */
    template<typename FloatType, auto dist = square_dist<FloatType>>
    [[nodiscard]] FloatType lb_Keogh(
            const FloatType *query, size_t length_query,
            const FloatType *upper, const FloatType *lower,
            FloatType ub
    ) {
        FloatType lb{0};

        for (size_t i = 0; i < length_query && lb < ub; i++) {
            FloatType qi{query[i]};
            if (const auto ui{upper[i]}; qi > ui) {
                lb += dist(qi, ui);
            } else if (const auto li{lower[i]}; qi < li) {
                lb += dist(qi, li);
            }
        }

        return lb;
    }


    template<typename FloatType, auto dist = square_dist<FloatType>>
    [[nodiscard]] inline FloatType lb_Keogh(
            const std::vector<FloatType> &query,
            const std::vector<FloatType> &upper,
            const std::vector<FloatType> &lower,
            FloatType ub
    ) {
        // lb_Keogh requires same size series
        assert(query.size() == upper.size());
        assert(query.size() == lower.size());
        return lb_Keogh<FloatType, dist>(query.data(), query.size(), upper.data(), lower.data(), ub);
    }

} // End of namespace tempo::univariate