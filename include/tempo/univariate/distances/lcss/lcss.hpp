#pragma once

#include "../../../tseries/tseries.hpp"
#include "../../../utils/utils.hpp"
#include "../distances.hpp"

namespace tempo::univariate {

    namespace internal {

        /// Check if two FloatType numbers are within EPSILON (1 = similar, 0 = not similar)
        template<typename FloatType>
        [[nodiscard]] bool sim(FloatType a, FloatType b, FloatType e) {
            return std::fabs(a - b) < e;
        }

    } // End of namespace internal


    /** Longest Common SubSequence (LCSS).
     *  Double buffered implementation using O(n) space.
     * @tparam FloatType    The floating number type used to represent the series.
     * @param series1   Pointer to the first series' values
     * @param length1   Length of the first series
     * @param series2   Pointer to the second series' values
     * @param length2   Length of the second series
     * @param epsilon   Threshold comparison for FloatType (consider v1 = v2 if |v1-v2|<epsilon)
     * @param w         Half-window parameter (looking at w cells on each side of the diagonal)
     *                  Must be 0<=w<=nblines and nblines - nbcols <= w
     * @return LCSS dissimilarity measure [0,1] where 0 stands for identical series and 1 completely distinct,
     *         or +INF if, given the w, no alignment is possible
     */
    template<typename FloatType>
    [[nodiscard]] FloatType lcss(
            const FloatType *series1, size_t length1,
            const FloatType *series2, size_t length2,
            FloatType epsilon,
            size_t w
    ) {
        const auto check_result = check_order_series(series1, length1, series2, length2);
        switch (check_result.index()) {
            case 0: { return std::get<0>(check_result); }
            case 1: {
                constexpr auto POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<FloatType>;
                // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                const auto[lines, nblines, cols, nbcols] = std::get<1>(check_result);
                // Cap the windows and check that, given the constraint, an alignment is possible
                if (w > nblines) { w = nblines; }
                if (nblines - nbcols > w) { return POSITIVE_INFINITY; }

                // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                // Double buffer allocation, no initialisation required (border condition manage in the code).
                // Base indices for the 'c'urrent row and the 'p'revious row. Account for the extra cell (+1 and +2)
                std::vector<FloatType> buffers_v((1 + nbcols) * 2, 0);
                FloatType *buffers = buffers_v.data();
                size_t c{0 + 1}, p{nbcols + 2};

                // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                // Initialisation: OK, border line and "first diag" init to 0

                // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                // Main loop
                for (size_t i{0}; i < nblines; ++i) {
                    // --- --- --- Swap and variables init
                    std::swap(c, p);
                    const double li = lines[i];
                    const size_t jStart = cap_start_index_to_window(i, w);
                    const size_t jStop = cap_stop_index_to_window_or_end(i, w, nbcols);
                    // --- --- --- Init the border (very first column)
                    buffers[c + jStart - 1] = 0;
                    // --- --- --- Iterate through the columns
                    for (size_t j{jStart}; j < jStop; ++j) {
                        if (internal::sim(li, cols[j], epsilon)) {
                            buffers[c + j] = buffers[p + j - 1] + 1; // Diag + 1
                        } else { // Note: Diagonal lookup required, e.g. when w=0
                            buffers[c + j] = max(buffers[c + j - 1], buffers[p + j - 1], buffers[p + j]);
                        }
                    }
                }

                // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                // Finalisation: put the result on a [0 - 1] range
                return 1.0 - (double(buffers[c + nbcols - 1]) / nbcols);
            }
            default: should_not_happen();
        }
    }


    /// Helper for the above, using vectors
    template<typename FloatType>
    [[nodiscard]] inline FloatType lcss(
            const std::vector<FloatType> &series1,
            const std::vector<FloatType> &series2,
            FloatType epsilon,
            size_t w) {
        return lcss<FloatType>(series1.data(), series1.size(), series2.data(), series2.size(), epsilon, w);
    }


    /// Helper for the above, using TSeries
    template<typename FloatType, typename LabelType>
    [[nodiscard]] inline FloatType lcss(
            const TSeries<FloatType, LabelType> &series1,
            const TSeries<FloatType, LabelType> &series2,
            FloatType epsilon,
            size_t w) {
        return lcss<FloatType>(series1.data(), series1.length(), series2.data(), series2.length(), epsilon, w);
    }

    /// Build a distfun_t for the above
    template<typename FloatType, typename LabelType>
    [[nodiscard]] inline distfun_t<FloatType, LabelType> distfun_lcss(FloatType epsilon, size_t w){
        return distfun_t<FloatType, LabelType> {
                [epsilon, w](
                        const TSeries<FloatType, LabelType>& series1,
                        const TSeries<FloatType, LabelType>& series2
                ){
                    return lcss<FloatType, LabelType>(series1, series2, epsilon, w);
                }
        };
    }


    // --- --- --- --- --- ---
    // LCSS with cut-off
    // --- --- --- --- --- ---

    /** Longest Common SubSequence (LCSS), with cut-off point for early abandoning.
     *  Double buffered implementation using O(n) space.
     * @tparam FloatType    The floating number type used to represent the series.
     * @param series1   Pointer to the first series' values
     * @param length1   Length of the first series
     * @param series2   Pointer to the second series' values
     * @param length2   Length of the second series
     * @param epsilon   Threshold comparison for FloatType (consider v1 = v2 if |v1-v2|<epsilon)
     * @param w         Half-window parameter (looking at w cells on each side of the diagonal)
     * @param cutoff.   Attempt to prune computation of alignments with cost > cutoff.
     *                  May lead to early abandoning.
     * @return LCSS dissimilarity measure [0,1] where 0 stands for identical series and 1 completely distinct,
     *         or +INF if early abandoned, or, given w, no alignment is possible
     */
    template<typename FloatType>
    [[nodiscard]] FloatType lcss(
            const FloatType *series1, size_t length1,
            const FloatType *series2, size_t length2,
            FloatType epsilon,
            size_t w,
            FloatType cutoff
    ) {
        const auto check_result = check_order_series(series1, length1, series2, length2);
        switch (check_result.index()) {
            case 0: { return std::get<0>(check_result); }
            case 1: {
                constexpr auto POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<FloatType>;
                // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                const auto[lines, nblines, cols, nbcols] = std::get<1>(check_result);
                // Cap the windows and check that, given the constraint, an alignment is possible
                if (w > nblines) { w = nblines; }
                if (nblines - nbcols > w) { return POSITIVE_INFINITY; }

                // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                // Double buffer allocation, init to 0.
                // Base indices for the 'c'urrent row and the 'p'revious row. Account for the extra cell (+1 and +2)
                std::vector<FloatType> buffers_v((1 + nbcols) * 2, 0);
                FloatType *buffers = buffers_v.data();
                size_t c{0 + 1}, p{nbcols + 2};

                // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                // Score to reach to equal ub, to beat to do better
                if (cutoff > 1) { cutoff = 1; }
                const size_t to_reach = std::floor((1 - cutoff) * nbcols);
                size_t current_max = 0;

                // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                // Initialisation: OK, border line and "first diag" init to 0

                // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                // Main loop
                for (size_t i{0}; i < nblines; ++i) {
                    // --- --- --- Stop if not enough remaining lines to reach the target (by taking the diagonal)
                    const size_t lines_left = nblines - i;
                    if (current_max + lines_left < to_reach) { return POSITIVE_INFINITY; }
                    // --- --- --- Swap and variables init
                    std::swap(c, p);
                    const FloatType li = lines[i];
                    const size_t jStart = cap_start_index_to_window(i, w);
                    const size_t jStop = cap_stop_index_to_window_or_end(i, w, nbcols);
                    // --- --- --- Init the border (very first column)
                    buffers[c + jStart - 1] = 0;
                    // --- --- --- Iterate through the columns
                    for (size_t j{jStart}; j < jStop; ++j) {
                        if (internal::sim(li, cols[j], epsilon)) {
                            const size_t cost = buffers[p + j - 1] + 1; // Diag + 1
                            current_max = std::max(current_max, cost);
                            buffers[c + j] = cost;
                        } else { // Note: Diagonal lookup required, e.g. when w=0
                            buffers[c + j] = max(buffers[c + j - 1], buffers[p + j - 1], buffers[p + j]);
                        }
                    }
                }

                // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                // Finalisation: put the result on a [0 - 1] range
                return 1.0 - (FloatType(buffers[c + nbcols - 1]) / nbcols);
            }
            default: should_not_happen();
        }
    }


    /// Helper for the above, using vectors
    template<typename FloatType>
    [[nodiscard]] inline FloatType lcss(
            const std::vector<FloatType> &series1,
            const std::vector<FloatType> &series2,
            FloatType epsilon,
            size_t w,
            FloatType cutoff) {
        return lcss<FloatType>(series1.data(), series1.size(), series2.data(), series2.size(), epsilon, w, cutoff);
    }


    /// Helper for the above, using TSeries
    template<typename FloatType, typename LabelType>
    [[nodiscard]] inline FloatType lcss(
            const TSeries<FloatType, LabelType> &series1,
            const TSeries<FloatType, LabelType> &series2,
            FloatType epsilon,
            size_t w,
            FloatType cutoff) {
        return lcss<FloatType>(series1.data(), series1.length(), series2.data(), series2.length(), epsilon, w, cutoff);
    }

    /// Build a distfun_cutoff_t for the above
    template<typename FloatType, typename LabelType>
    [[nodiscard]] inline distfun_cutoff_t<FloatType, LabelType> distfun_cutoff_lcss(FloatType epsilon, size_t w) {
        return distfun_cutoff_t<FloatType, LabelType>{
                [epsilon, w](
                        const TSeries<FloatType, LabelType> &series1,
                        const TSeries<FloatType, LabelType> &series2,
                        FloatType co
                ) {
                    return lcss<FloatType, LabelType>(series1, series2, epsilon, w, co);
                }
        };
    }

} // End of namespace tempo::univariate
