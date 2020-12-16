#pragma once

#include "../../../utils/utils.hpp"
#include "../distances.hpp"

namespace tempo::univariate {

    namespace internal {

        /** Move Split Merge metric with cutoff point for early abandoning and pruning.
         *  Double buffered implementation using O(n) space.
         *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning).
         *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
         *  Actual implementation assuming that some pre-conditions are fulfilled.
         * @tparam FloatType    The floating number type used to represent the series.
         * @param lines     Pointer to the "line series". Must be the longest series. Cannot be null.
         * @param nblines   Length of the line series. Must be 0 < nbcols <= nblines < tempo::MAX_SERIES_LENGTH.
         * @param cols      Pointer to the "column series". Must be the shortest series. Cannot be null.
         * @param nbcols    Length of the column series. Must be 0 < nbcols <= nblines < tempo::MAX_SERIES_LENGTH.
         * @param co        Cost of the Split and Merge operations, also added to Move operations. Must have c>=0.
         * @param cutoff.   Attempt to prune computation of alignments with cost > cutoff.
         *                  May lead to early abandoning.
         * @return  DTW between the two series or +INF if early abandoned.
         *          Warning: a result different from +INF does not warrant a cost < cutoff.
         */
        template<typename FloatType = double>
        [[nodiscard]] FloatType msm(
                const FloatType *lines, size_t nblines,
                const FloatType *cols, size_t nbcols,
                const FloatType co,
                FloatType cutoff
        ) {
            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            // In debug mode, check preconditions
            assert(lines != nullptr && nblines != 0 && nblines < MAX_SERIES_LENGTH);
            assert(cols != nullptr && nbcols != 0 && nbcols < MAX_SERIES_LENGTH);
            assert(nbcols <= nblines);
            assert(w <= nblines);
            assert(nblines-nbcols<=w);
            // Adapt constants to the floating point type
            constexpr auto POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<FloatType>;

            return POSITIVE_INFINITY;
        }

    } // End of namespace internal

    // --- --- --- --- ---
    // --- MSM
    // --- --- --- --- ---

    /** Move Split Merge metric.
     *  Double buffered implementation using O(n) space.
     *  Compute an upper bound before calling the function with pruning.
     *  Any valid path in the cost matrix represent an upper bound.
     *  We compute such a path using the diagonal, then going down in the last column when lengths are disparate.
     * @tparam FloatType    The floating number type used to represent the series.
     * @param series1   Pointer to the first series' values.
     * @param length1   Length of the first series. Must be < tempo::MAX_SERIES_LENGTH.
     * @param series2   Pointer to the second series' values.
     * @param length2   Length of the second series. Must be < tempo::MAX_SERIES_LENGTH.
     * @param co        Cost of the Split and Merge operations, also added to Move operations. Must have c>=0.
     * @return  DTW between the two series
     */
    template<typename FloatType = double>
    [[nodiscard]] FloatType msm(
            const FloatType *series1, size_t length1,
            const FloatType *series2, size_t length2,
            const FloatType co
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

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Compute a cutoff point using the diagonal
        FloatType cutoff{POSITIVE_INFINITY};
        //// Counter, will first go over the columns, and then complete the lines
        //size_t i{0};
        //// We have less columns than lines: cover all the columns first.
        //for (; i < nbcols; ++i) { cutoff += dist(lines[i], cols[i]); }
        //// Then go down in the last column
        //if(i<nblines) {
        //    const auto lc = cols[nbcols - 1];
        //    for (; i < nblines; ++i) { cutoff += dist(lines[i], lc); }
        //}

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        return internal::msm<FloatType>(lines, nblines, cols, nbcols, co, cutoff);
    }

    /// Helper for the above, using vectors
    template<typename FloatType=double>
    [[nodiscard]] inline FloatType msm(
            const std::vector<FloatType>& series1,
            const std::vector<FloatType>& series2,
            const FloatType co){
        return msm<FloatType>(USE(series1), USE(series2), co);
    }

    // --- --- --- --- ---
    // --- MSM with cutoff
    // --- --- --- --- ---

    /** Move Split Merge metric with cutoff point for early abandoning and pruning.
     *  Double buffered implementation using O(n) space.
     *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning).
     *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
     * @tparam FloatType    The floating number type used to represent the series.
     * @param series1   Pointer to the first series' values.
     * @param length1   Length of the first series. Must be < tempo::MAX_SERIES_LENGTH.
     * @param series2   Pointer to the second series' values.
     * @param length2   Length of the second series. Must be < tempo::MAX_SERIES_LENGTH.
     * @param co        Cost of the Split and Merge operations, also added to Move operations. Must have c>=0.
     * @param cutoff.   Attempt to prune computation of alignments with cost > cutoff.
     *                  May lead to early abandoning.
     * @return  DTW between the two series or +INF if early abandoned.
     *          Warning: a result different from +INF does not warrant a cost < cutoff.
     */
    template<typename FloatType = double>
    [[nodiscard]] FloatType msm(
            const FloatType *series1, size_t length1,
            const FloatType *series2, size_t length2,
            const FloatType co,
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

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        return internal::msm<FloatType>(lines, nblines, cols, nbcols, co, cutoff);
    }

    /// Helper for the above, using vectors
    template<typename FloatType=double>
    [[nodiscard]] inline FloatType msm(
            const std::vector<FloatType>& series1,
            const std::vector<FloatType>& series2,
            const FloatType co,
            FloatType cutoff){
        return msm<FloatType>(USE(series1), USE(series2), co, cutoff);
    }

} // End of namespace tempo::univariate
