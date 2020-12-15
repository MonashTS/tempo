#pragma once

#include "../../../utils/utils.hpp"
#include "../distances.hpp"

namespace tempo::univariate::distances {

    namespace internal {

        /// Check if two FloatType numbers are within EPSILON (1 = similar, 0 = not similar)
        template<typename FloatType=double>
        [[nodiscard]] bool sim(FloatType a, FloatType b, FloatType e) {
            return std::fabs(a - b) < e;
        }

    } // End of namespace internal


    /** LCSS (Longest Common SubSequence) on a FloatType buffer.
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
    template<typename FloatType=double>
    [[nodiscard]] FloatType lcss(
            const FloatType *series1, size_t length1,
            const FloatType *series2, size_t length2,
            FloatType epsilon,
            size_t w
    ) {
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        constexpr auto POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<FloatType>;
        // Pre-conditions. Accept nullptr if length is 0
        assert((series1 != nullptr || length1 == 0) && length1 < MAX_SERIES_LENGTH);
        assert((series2 != nullptr || length2 == 0) && length2 < MAX_SERIES_LENGTH);
        // Check sizes. If both series are empty, return 0, else if one is empty and not the other, maximal error.
        if (length1 == 0 && length2 == 0) { return 0; }
        else if ((length1 == 0) != (length2 == 0)) { return POSITIVE_INFINITY; }
        // Use the smallest size as the columns (which will be the allocation size)
        const auto[lines, nblines, cols, nbcols] =
        (length1 > length2) ?
        std::tuple(series1, length1, series2, length2) : std::tuple(series2, length2, series1, length1);
        // Cap the windows and check that, given the constraint, an alignment is possible
        if (w > nblines) { w = nblines; }
        if (nblines - nbcols > w) { return POSITIVE_INFINITY; }

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Double buffer allocation, no initialisation required (border condition manage in the code).
        // Base indices for the 'c'urrent row and the 'p'revious row. Account for the extra cell (+1 and +2)
        std::vector<FloatType> buffers_v((1 + nbcols) * 2, 0);
        FloatType *buffers = buffers_v.data();
        size_t c{0 + 1}, p{nbcols + 2};

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Initialisation: OK, border line and "first diag" init to 0

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
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

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Finalisation: put the result on a [0 - 1] range
        return 1.0 - (double(buffers[c + nbcols - 1]) / nbcols);
    }


    /// Helper for the above, using vectors
    template<typename FloatType=double>
    [[nodiscard]] inline FloatType lcss(
            const std::vector<FloatType> &series1,
            const std::vector<FloatType> &series2,
            FloatType epsilon, size_t w) {
        return lcss<FloatType>(USE(series1), USE(series2), epsilon, w);
    }


    // --- --- --- --- --- ---
    // LCSS with cut-off
    // --- --- --- --- --- ---

    /** LCSS (Longest Common SubSequence) on a FloatType buffer.
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
     *         or +INF if, given w, no alignment is possible
     */
    template<typename FloatType=double>
    [[nodiscard]] FloatType lcss(
            const FloatType *series1, size_t length1,
            const FloatType *series2, size_t length2,
            FloatType epsilon,
            size_t w,
            FloatType cutoff
    ) {
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        constexpr auto POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<FloatType>;
        // Pre-conditions. Accept nullptr if length is 0
        assert((series1 != nullptr || length1 == 0) && length1 < MAX_SERIES_LENGTH);
        assert((series2 != nullptr || length2 == 0) && length2 < MAX_SERIES_LENGTH);
        // Check sizes. If both series are empty, return 0, else if one is empty and not the other, maximal error.
        if (length1 == 0 && length2 == 0) { return 0; }
        else if ((length1 == 0) != (length2 == 0)) { return POSITIVE_INFINITY; }
        // Use the smallest size as the columns (which will be the allocation size)
        const auto[lines, nblines, cols, nbcols] =
        (length1 > length2) ?
        std::tuple(series1, length1, series2, length2) : std::tuple(series2, length2, series1, length1);
        // Cap the windows and check that, given the constraint, an alignment is possible
        if (w > nblines) { w = nblines; }
        if (nblines - nbcols > w) { return POSITIVE_INFINITY; }

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Double buffer allocation, no initialisation required (border condition manage in the code).
        // Base indices for the 'c'urrent row and the 'p'revious row. Account for the extra cell (+1 and +2)
        std::vector<FloatType> buffers_v((1 + nbcols) * 2, 0);
        FloatType *buffers = buffers_v.data();
        size_t c{0 + 1}, p{nbcols + 2};

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Score to reach to equal ub, to beat to do better
        if (cutoff > 1) { cutoff = 1; }
        const size_t to_reach = std::floor((1 - cutoff) * nbcols);
        size_t current_max = 0;

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Initialisation: OK, border line and "first diag" init to 0

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
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

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Finalisation: put the result on a [0 - 1] range
        return 1.0 - (FloatType(buffers[c + nbcols - 1]) / nbcols);
    }


    /// Helper for the above, using vectors
    template<typename FloatType=double>
    [[nodiscard]] inline FloatType lcss(
            const std::vector<FloatType> &series1,
            const std::vector<FloatType> &series2,
            FloatType epsilon, size_t w, FloatType cutoff) {
        return lcss<FloatType>(USE(series1), USE(series2), epsilon, w, cutoff);
    }


} // End of namespace tempo::univariate::distances
