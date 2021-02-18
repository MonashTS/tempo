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



    template<typename FloatType, auto dist = square_dist<FloatType>>
    [[nodiscard]] FloatType lb_Keogh2j(
            const FloatType *query, size_t lq,
            const FloatType *qu, const FloatType *ql,
            const FloatType *candidate, [[maybe_unused]] size_t lc,
            const FloatType *cu, const FloatType *cl,
            FloatType ub
    ) {
        // Init
        FloatType lb1{0};
        FloatType lb2{0};
        // Main loop, continue while we are <= ub
        for (size_t i = 0; i < lq && lb1 <= ub && lb2 <= ub; ++i) {
            // Query - envelope candidate
            {
                FloatType qi{query[i]};
                if (const auto ui{cu[i]}; qi > ui) { lb1 += dist(qi, ui); }
                else if (const auto li{cl[i]}; qi < li) { lb1 += dist(qi, li); }
            }
            // Candidate - envelope query
            {
                FloatType ci{candidate[i]};
                if (const auto ui{qu[i]}; ci > ui) { lb2 += dist(ci, ui); }
                else if (const auto li{ql[i]}; ci < li) { lb2 += dist(ci, li); }
            }
        }
        FloatType lb = std::max<FloatType>(lb1, lb2);
        return (lb>ub)?POSITIVE_INFINITY<FloatType>:lb;
    }



    /** Dataset/TSPack transformer for LB Keogh.
     *  Input: TSeries<FloatType, LabelType>
     *  Output: A tuple of vector<FloatType> containing the envelopes (Upper, Lower).
     *  Use the provided static functions for easier access.
     */
    template<typename FloatType, typename LabelType>
    struct KeoghEnvelopesTransformer {
        static constexpr auto name = "keogh_envelopes";
        using Vec = std::vector<FloatType>;
        using ElemType = std::tuple<Vec,Vec>;
        using TS = TSeries<FloatType, LabelType>;
        using TSP = TSPack<FloatType, LabelType>;
        using TSPTr = TSPackTransformer<FloatType, LabelType>;

        [[nodiscard]] static TSPTr get(size_t w, size_t source_index, const std::string& pfx){
            auto n = pfx + "_" + name;
            return TSPTr {
                    .name = n,
                    .extra_json = "{\"window\":" + std::to_string(w) + "}",
                    .transfun = [source_index, n, w](const TSPack<FloatType, LabelType>& tsp){
                        const auto& s = *(static_cast<TS*>(tsp.transforms[source_index]));
                        std::vector<FloatType> upper(s.size());
                        std::vector<FloatType> lower(s.size());
                        get_keogh_envelopes(s.data(), s.size(), upper.data(), lower.data(), w);
                        auto capsule = make_capsule<ElemType>(std::move(upper), std::move(lower));
                        auto* ptr = capsule_ptr<ElemType>(capsule);
                        return TSPackResult {
                                TSPackTR {
                                        .name = n,
                                        .capsule = capsule,
                                        .transform = ptr
                                }
                        };
                    }
            };
        }

        [[nodiscard]] inline static const ElemType& cast(void* ptr){
            return *(static_cast<ElemType*>(ptr));
        }

        [[nodiscard]] inline static const Vec& up(void* ptr){
            return std::get<0>(cast(ptr));
        }

        [[nodiscard]] inline static const Vec& lo(void* ptr){
            return std::get<1>(cast(ptr));
        }
    };

} // End of namespace tempo::univariate