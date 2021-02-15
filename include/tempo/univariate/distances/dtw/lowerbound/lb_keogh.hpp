#pragma once

#include "../../../../utils/utils.hpp"
#include "../../distances.hpp"

namespace tempo::univariate {

    template<typename FloatType, auto dist = square_dist<FloatType>>
    [[nodiscard]] FloatType lb_Keogh(
            const FloatType *query, size_t lq,
            const FloatType *upper, const FloatType *lower, size_t lc,
            size_t w,
            FloatType ub
    ) {
        // Pre check - Empty series
        if (lq == 0 && lc == 0) { return {FloatType(0.0)}; }
        else if ((lq == 0) != (lc == 0)) { return POSITIVE_INFINITY<FloatType>; }

        // Pre check - Diff length
        if(lq==lc){
            // Init
            FloatType lb{0};
            // Main loop, continue while we are <= ub
            for (size_t i = 0; i < lq && lb <= ub; ++i) {
                FloatType qi{query[i]};
                if (const auto ui{upper[i]}; qi > ui) { lb += dist(qi, ui); }
                else if (const auto li{lower[i]}; qi < li) { lb += dist(qi, li); }
            }
            return (lb>ub)?POSITIVE_INFINITY<FloatType>:lb;
        } else {
            // Pre check - does the window allow an alignment
            const auto lmax = std::max<size_t>(lq, lc);
            const auto lmin = std::min<size_t>(lq, lc);
            if (w > lmax) { w = lmax; }
            if (lmax - lmin > w) { return POSITIVE_INFINITY<FloatType>; }
            // Init
            FloatType lb{0};
            // Main loop, continue while we are <= ub
            for (size_t i = 0; i < lmin && lb <= ub; ++i) {
                FloatType qi{query[i]};
                if (const auto ui{upper[i]}; qi > ui) { lb += dist(qi, ui); }
                else if (const auto li{lower[i]}; qi < li) { lb += dist(qi, li); }
            }
            // Secondary loop, only if the smallest series is the candidate
            if (lmin == lc) {
                const auto ui = upper[lmin-1];
                const auto li = lower[lmin-1];
                for(size_t i=lmin; i<lmax && lb<=ub; ++i){
                    FloatType qi{query[i]};
                    if (qi > ui) { lb += dist(qi, ui); }
                    else if (qi < li) { lb += dist(qi, li); }
                }
            }
            return (lb>ub)?POSITIVE_INFINITY<FloatType>:lb;
        }
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