#pragma once

#include "../../../../utils/utils.hpp"
#include "../../distances.hpp"
#include "envelopes.hpp"

namespace tempo::univariate {

    /** LB_Enhanced, early stopping if above ub. Series must have the same length!
     * @param query             The time series being queried
     * @param lq                Length of query
     * @param candidate         Time series from the database
     * @param lc                Length of candidate
     * @param upper             Upper envelope of candidate
     * @param lower             Lower envelope of candidate
     * @param v                 Speed/Tightness trade-off (faster = 0, tighter = min(llq, lc)/2)
     * @param w                 Warping window
     * @param cutoff            Cutoff point: early abandon if computed value > cutoff
     * @return
     */
    template<typename FloatType, auto dist = square_dist<FloatType>>
    [[nodiscard]] FloatType lb_Enhanced(
            const FloatType *query, size_t lq,
            const FloatType *candidate, [[maybe_unused]] size_t lc,
            const FloatType *upper, const FloatType *lower,
            size_t v,
            size_t w,
            FloatType cutoff
    ) {
        FloatType lb{INITLB<FloatType>}; // Init with a small negative value: handle numerical instability
        size_t nbands = std::min(lq/2, v);
        // --- --- --- Do L & R Bands
        // First alignment
        lb += dist(query[0], candidate[0]);
        // Manage the case of series of length 1
        if (lq == 1) { return (lb > cutoff) ? POSITIVE_INFINITY<FloatType> : lb; }
        const size_t last = lq - 1;
        // Last alignment
        lb += dist(query[last], candidate[last]);
        // L & R bands
        for (size_t i = 1; i < nbands && lb <= cutoff; ++i) {
            const auto fixR = last-i;
            FloatType minL = dist(query[i], candidate[i]);
            FloatType minR = dist(query[fixR], candidate[fixR]);
            for (size_t j = cap_start_index_to_window(i, w); j < i; ++j) {
                const auto movR = last-j;
                minL = min(minL, dist(query[i], candidate[j]), dist(query[j], candidate[i]));
                minR=min(minR, dist(query[fixR], candidate[movR]), dist(query[movR], candidate[fixR]));
            }
            lb = lb + minL + minR;
        }
        // --- --- ---
        if(lb>cutoff){return POSITIVE_INFINITY<FloatType>;}

        // --- --- --- Bridge with LB Keogh, continue while we are <= cutoff
        const auto end = lq-nbands;
        for (size_t i = nbands; i < end && lb <= cutoff; ++i) {
            FloatType qi{query[i]};
            if (const auto ui{upper[i]}; qi > ui) { lb += dist(qi, ui); }
            else if (const auto li{lower[i]}; qi < li) { lb += dist(qi, li); }
        }

        return (lb>cutoff)?POSITIVE_INFINITY<FloatType>:lb;
    }



    template<typename FloatType, auto dist = square_dist<FloatType>>
    [[nodiscard]] FloatType lb_Enhanced2(
            const FloatType *query, size_t lq,
            const FloatType *qup, const FloatType *qlo,
            const FloatType *candidate, [[maybe_unused]] size_t lc,
            const FloatType *cup, const FloatType *clo,
            size_t v,
            size_t w,
            FloatType cutoff
    ) {
        FloatType lb1{INITLB<FloatType>}; // Init with a small negative value: handle numerical instability
        size_t nbands = std::min(lq/2, v);
        // --- --- --- Do L & R Bands
        // First alignment
        lb1 += dist(query[0], candidate[0]);
        // Manage the case of series of length 1
        if (lq == 1) { return (lb1 > cutoff) ? POSITIVE_INFINITY<FloatType> : lb1; }
        const size_t last = lq - 1;
        // Last alignment
        lb1 += dist(query[last], candidate[last]);
        // L & R bands
        for (size_t i = 1; i < nbands && lb1 <= cutoff; ++i) {
            const auto fixR = last-i;
            FloatType minL = dist(query[i], candidate[i]);
            FloatType minR = dist(query[fixR], candidate[fixR]);
            for (size_t j = cap_start_index_to_window(i, w); j < i; ++j) {
                const auto movR = last-j;
                minL = min(minL, dist(query[i], candidate[j]), dist(query[j], candidate[i]));
                minR=min(minR, dist(query[fixR], candidate[movR]), dist(query[movR], candidate[fixR]));
            }
            lb1 = lb1 + minL + minR;
        }
        // --- --- ---
        if(lb1 > cutoff){return POSITIVE_INFINITY<FloatType>;}

        // --- --- --- Bridge with LB Keogh, continue while we are <= cutoff
        FloatType lb2{lb1};
        const auto end = lq-nbands;
        for (size_t i = nbands; i < end && lb1 <= cutoff && lb2 <= cutoff; ++i) {
            // Query - envelope candidate
            {
                FloatType qi{query[i]};
                if (const auto ui{cup[i]}; qi > ui) { lb1 += dist(qi, ui); }
                else if (const auto li{clo[i]}; qi < li) { lb1 += dist(qi, li); }
            }
            // Candidate - envelope query
            {
                FloatType ci{candidate[i]};
                if (const auto ui{qup[i]}; ci > ui) { lb2 += dist(ci, ui); }
                else if (const auto li{qlo[i]}; ci < li) { lb2 += dist(ci, li); }
            }
        }

        lb1=std::max<FloatType>(lb1, lb2);
        return (lb1 > cutoff) ? POSITIVE_INFINITY<FloatType> : lb1;
    }






        template<typename FloatType, auto dist = square_dist<FloatType>>
    [[nodiscard]] inline FloatType lb_Enhanced(
            const std::vector<FloatType> &query,
            const std::vector<FloatType> &candidate,
            const std::vector<FloatType> &upper,
            const std::vector<FloatType> &lower,
            size_t v,
            size_t w,
            FloatType cutoff
    ){
        assert(query.size() == candidate.size());
        assert(candidate.size() == upper.size());
        assert(candidate.size() == lower.size());
        return lb_Enhanced(query.data(), query.size(), candidate.data(),
                           candidate.size(), upper.data(), lower.data(), v, w, cutoff);
    }

} // End of namespace tempo::univariate