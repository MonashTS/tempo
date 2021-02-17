#pragma once

#include "../../../../tseries/tseries.hpp"
#include "../../../../tseries/tspack.hpp"
#include "../../../../utils/utils.hpp"

#include <deque>

namespace tempo::univariate {

    /** Compute the upper and lower envelopes of a series, suitable for LB_Keogh
    *  Implementation based on Lemire's method
    *    Lemire D (2009) Faster retrieval with a two-pass dynamic-time-warping lower bound.
    *    Pattern Recognition 42:2169–2180. https://doi.org/10.1016/j.patcog.2008.11.030
    * @param series Input series
    * @param length Length of the input series
    * @param upper Output array for the upper envelope - Must be able to store 'length' element
    * @param lower Output array for the lower envelope - Must be able to store 'length' element
    * @param w The window for which the envelope is computed.
    */
    template<typename FloatType>
    void get_keogh_envelopes(
            const FloatType *series, size_t length,
            FloatType *upper,
            FloatType *lower,
            size_t w) {

        // --- Window size adjustment and early exit
        if (length == 0) { return; }
        if (w >= length) { w = length - 1; }
        if (w == 0) {
            for (size_t i = 0; i < length; ++i) { upper[i] = lower[i] = series[i]; }
            return;
        }

        // --- Initialize the queues with the first w points
        std::deque<size_t> up{0};   // Contains indexes of decreasing values series[idx] (done with (1)). front is max
        std::deque<size_t> lo{0};   // Contains indexes of increasing values series[idx] (done with (2)). front is min
        for (size_t i{1}; i < w; ++i) {
            const FloatType prev{series[i - 1]};
            const FloatType si{series[i]};
            // remark comparison or strict comparison does not matters, hence the else allow to avoid an extra if
            if (prev <= si) {
                do { up.pop_back(); }
                while (!up.empty() && series[up.back()] <= si);
            } // (1) Remove while si is larger than up[back]
            else {
                do { lo.pop_back(); }
                while (!lo.empty() && series[lo.back()] >= si);
            }        // (2) Remove while si is smaller than lo[back]
            up.push_back(i);
            lo.push_back(i);
        }

        // --- Go over the series up to length-(w+1)
        // update queue[i+w+1], then update envelopes[i] with front of the queue
        size_t up_front_idx{up.front()};
        FloatType up_front_val{series[up_front_idx]};
        size_t lo_front_idx{lo.front()};
        FloatType lo_front_val{series[lo_front_idx]};
        for (size_t i{0}; i < length - w; ++i) {
            // Update the queues:
            const size_t idx{i + w};
            const FloatType prev{series[idx - 1]}; // Ok as w > 0
            const FloatType si{series[idx]};
            // 1) Evict item preventing monotonicity.
            // If a queue is empty, the item to add is also the new front
            if (prev <= si) {
                do { up.pop_back(); } while (!up.empty() && series[up.back()] <= si);
                if (up.empty()) {
                    up_front_idx = idx;
                    up_front_val = series[up_front_idx];
                }
            } else {
                do { lo.pop_back(); } while (!lo.empty() && series[lo.back()] >= si);
                if (lo.empty()) {
                    lo_front_idx = idx;
                    lo_front_val = series[lo_front_idx];
                }
            }
            // 2) Push back index, then update envelopes with front indexes
            up.push_back(idx);
            lo.push_back(idx);
            upper[i] = up_front_val; // max over range
            lower[i] = lo_front_val; // min over range
            // 3) trim the front
            if (up_front_idx + w <= i) {
                up.pop_front();
                up_front_idx = up.front();
                up_front_val = series[up_front_idx];
            }
            if (lo_front_idx + w <= i) {
                lo.pop_front();
                lo_front_idx = lo.front();
                lo_front_val = series[lo_front_idx];
            }
        }

        // --- Finish the last w+1 items: values are present already in up_front_val and lo_front_val
        for (size_t i{length - w}; i < length; ++i) {
            // Update the envelope
            upper[i] = up_front_val;
            lower[i] = lo_front_val;
            // Trim the front
            if (up_front_idx + w <= i) {
                up.pop_front();
                up_front_idx = up.front();
                up_front_val = series[up_front_idx];
            }
            if (lo_front_idx + w <= i) {
                lo.pop_front();
                lo_front_idx = lo.front();
                lo_front_val = series[lo_front_idx];
            }
        }
    }

    /** Compute the upper and lower envelopes of a series, suitable for LB_Keogh.
     *  Wrapper for get_envelopes with vector
     * @param series Constant input series
     * @param upper Output series - reallocation may occur!
     * @param lower Output series - reallocation may occur!
     * @param w The window for which the envelope is computed.
     */
    template<typename FloatType>
    inline void get_keogh_envelopes(const std::vector<FloatType> &series,
                              std::vector<FloatType> &upper, std::vector<FloatType> &lower,
                              size_t w) {
        // Ensure the output vectors are large enough
        upper.resize(series.size());
        lower.resize(series.size());
        // Do the work
        get_keogh_envelopes(series.data(), series.size(), upper.data(), lower.data(), w);
    }


    /** Compute only the upper envelopes of a series.
     *  Implementation based on Lemire's method
     *    Lemire D (2009) Faster retrieval with a two-pass dynamic-time-warping lower bound.
     *    Pattern Recognition 42:2169–2180. https://doi.org/10.1016/j.patcog.2008.11.030
     * @param series Input series
     * @param length Length of the input series
     * @param upper Output array for the upper envelope - Must be able to store 'length' element
     * @param w The window for which the envelope is computed.
     */
    template<typename FloatType>
    void get_keogh_up_envelope(
            const FloatType *series, size_t length,
            FloatType *upper,
            size_t w) {

        // --- Window size adjustment and early exit
        if (length == 0) { return; }
        if (w >= length) { w = length - 1; }
        if (w == 0) {
            for (size_t i = 0; i < length; ++i) { upper[i] = series[i]; }
            return;
        }

        // --- Initialize the queues with the first w points
        std::deque<size_t> up{0};
        for (size_t i{1}; i < w; ++i) {
            const FloatType prev{series[i - 1]};
            const FloatType si{series[i]};
            if (prev <= si) { do { up.pop_back(); } while (!up.empty() && series[up.back()] <= si); }
            up.push_back(i);
        }

        // --- Go over the series up to length-(w+1)
        // update queue[i+w+1], then update envelopes[i] with front of the queue
        size_t up_front_idx{up.front()};
        FloatType up_front_val{series[up_front_idx]};
        for (size_t i{0}; i < length - w; ++i) {
            // Update the queues:
            const size_t idx{i + w};
            const FloatType prev{series[idx - 1]}; // Ok as w > 0
            const FloatType si{series[idx]};
            // 1) Evict item preventing monotonicity.
            // If a queue is empty, the item to add is also the new front
            if (prev <= si) {
                do { up.pop_back(); } while (!up.empty() && series[up.back()] <= si);
                if (up.empty()) {
                    up_front_idx = idx;
                    up_front_val = series[up_front_idx];
                }
            }
            // 2) Push back index, then update envelopes with front indexes
            up.push_back(idx);
            upper[i] = up_front_val;
            // 3) trim the front
            if (up_front_idx + w <= i) {
                up.pop_front();
                up_front_idx = up.front();
                up_front_val = series[up_front_idx];
            }
        }

        // --- Finish the last w+1 items: values are present already in up_front_val and lo_front_val
        for (size_t i{length - w}; i < length; ++i) {
            // Update the envelope
            upper[i] = up_front_val;
            // Trim the front
            if (up_front_idx + w <= i) {
                up.pop_front();
                up_front_idx = up.front();
                up_front_val = series[up_front_idx];
            }
        }
    }

    /** Compute only the upper envelopes of a series.
     *  Wrapper for get_envelopes with vector
     * @param series Constant input series
     * @param upper Output series - reallocation may occur!
     * @param w The window for which the envelope is computed.
     */
    template<typename FloatType>
    inline void get_keogh_up_envelope(const std::vector<FloatType> &series,
                                std::vector<FloatType> &upper,
                                size_t w) {
        // Ensure the output vector is large enough
        upper.resize(series.size());
        // Do the work
        get_up_envelope(series.data(), series.size(), upper.data(), w);
    }


    /** Compute lower envelopes of a series.
     *  Implementation based on Lemire's method
     *    Lemire D (2009) Faster retrieval with a two-pass dynamic-time-warping lower bound.
     *    Pattern Recognition 42:2169–2180. https://doi.org/10.1016/j.patcog.2008.11.030
     * @param series Input series
     * @param length Length of the input series
     * @param lower Output array for the lower envelope - Must be able to store 'length' element
     * @param w The window for which the envelope is computed.
     */
    template<typename FloatType>
    void get_keogh_lo_envelope(
            const FloatType *series, size_t length,
            FloatType *lower,
            size_t w) {

        // --- Window size adjustment and early exit
        if (length == 0) { return; }
        if (w >= length) { w = length - 1; }
        if (w == 0) {
            for (size_t i = 0; i < length; ++i) { lower[i] = series[i]; }
            return;
        }

        // --- Initialize the queues with the first w points
        std::deque<size_t> lo{0};   // Contains indexes of increasing values series[idx] (done with (2)). front is min
        for (size_t i{1}; i < w; ++i) {
            const FloatType prev{series[i - 1]};
            const FloatType si{series[i]};
            // remark comparison or strict comparison does not matters, hence the else allow to avoid an extra if
            if (prev >= si) { do { lo.pop_back(); } while (!lo.empty() && series[lo.back()] >= si); }
            lo.push_back(i);
        }

        // --- Go over the series up to length-(w+1)
        // update queue[i+w+1], then update envelopes[i] with front of the queue
        size_t lo_front_idx{lo.front()};
        FloatType lo_front_val{series[lo_front_idx]};
        for (size_t i{0}; i < length - w; ++i) {
            // Update the queues:
            const size_t idx{i + w};
            const FloatType prev{series[idx - 1]}; // Ok as w > 0
            const FloatType si{series[idx]};
            // 1) Evict item preventing monotonicity.
            // If a queue is empty, the item to add is also the new front
            if (prev >= si) {
                do { lo.pop_back(); } while (!lo.empty() && series[lo.back()] >= si);
                if (lo.empty()) {
                    lo_front_idx = idx;
                    lo_front_val = series[lo_front_idx];
                }
            }
            // 2) Push back index, then update envelopes with front indexes
            lo.push_back(idx);
            lower[i] = lo_front_val; // min over range
            // 3) trim the front
            if (lo_front_idx + w <= i) {
                lo.pop_front();
                lo_front_idx = lo.front();
                lo_front_val = series[lo_front_idx];
            }
        }

        // --- Finish the last w+1 items: values are present already in up_front_val and lo_front_val
        for (size_t i{length - w}; i < length; ++i) {
            // Update the envelope
            lower[i] = lo_front_val;
            // Trim the front
            if (lo_front_idx + w <= i) {
                lo.pop_front();
                lo_front_idx = lo.front();
                lo_front_val = series[lo_front_idx];
            }
        }
    }

    /** Compute the lower envelopes of a series.
     *  Wrapper for get_envelopes with vector
     * @param series Constant input series
     * @param lower Output series - reallocation may occur!
     * @param w The window for which the envelope is computed.
     */
    template<typename FloatType>
    inline void get_keogh_lo_envelope(const std::vector<FloatType> &series,
                                std::vector<FloatType> &lower,
                                size_t w) {
        // Ensure the output vector is large enough
        lower.resize(series.size());
        // Do the work
        get_lo_envelope(series.data(), series.size(), lower.data(), w);
    }

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

    template<typename FloatType, typename LabelType>
    struct KeoghEnvUPTransformer {
        static constexpr auto name = "keogh_env_up";
        using Vec = std::vector<FloatType>;
        using ElemType = Vec;
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
                        get_keogh_up_envelope(s.data(), s.size(), upper.data(), w);
                        auto capsule = make_capsule<ElemType>(std::move(upper));
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
    };

    template<typename FloatType, typename LabelType>
    struct KeoghEnvLOTransformer {
        static constexpr auto name = "keogh_env_lo";
        using Vec = std::vector<FloatType>;
        using ElemType = Vec;
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
                        std::vector<FloatType> lower(s.size());
                        get_keogh_lo_envelope(s.data(), s.size(), lower.data(), w);
                        auto capsule = make_capsule<ElemType>(std::move(lower));
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
    };


} // End of namespace tempo::univariate