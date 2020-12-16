#pragma once

#include "../../../utils/utils.hpp"
#include "../distances.hpp"

namespace tempo::univariate {

    namespace internal {

        /** Constrained Dynamic Time Warping with cutoff point for early abandoning and pruning.
         *  Double buffered implementation using O(n) space.
         *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning, large window).
         *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
         *  Actual implementation assuming that some pre-conditions are fulfilled.
         * @tparam FloatType    The floating number type used to represent the series.
         * @tparam dist     Distance function, default to square euclidean distance for FloatType.
         * @param lines     Pointer to the "line series". Must be the longest series. Cannot be null.
         * @param nblines   Length of the line series. Must be 0 < nbcols <= nblines < tempo::MAX_SERIES_LENGTH.
         * @param cols      Pointer to the "column series". Must be the shortest series. Cannot be null.
         * @param nbcols    Length of the column series. Must be 0 < nbcols <= nblines < tempo::MAX_SERIES_LENGTH.
         * @param w         Half-window parameter (looking at w cells on each side of the diagonal)
         *                  Must be 0<=w<=nblines and nblines - nbcols <= w
         * @param cutoff.   Attempt to prune computation of alignments with cost > cutoff.
         *                  May lead to early abandoning.
         * @return  DTW between the two series or +INF if early abandoned.
         *          Warning: a result different from +INF does not warrant a cost < cutoff.
         */
        template<typename FloatType = double, auto dist = square_dist < FloatType>>

        [[nodiscard]] inline FloatType cdtw(
                const FloatType *lines, size_t nblines,
                const FloatType *cols, size_t nbcols,
                const size_t w,
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

            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            // Create a new tighter upper bounds (most commonly used in the code).
            // First, take the "next float" after "cutoff" to deal with numerical instability.
            // Then, subtract the cost of the last alignment.
            const FloatType ub = nextafter(cutoff, POSITIVE_INFINITY) - dist(lines[nblines - 1], cols[nbcols - 1]);

            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            // Double buffer allocation, init to +INF.
            // Base indices for the 'c'urrent row and the 'p'revious row. Account for the extra cell (+1 and +2)
            std::vector<FloatType> buffers_v((1 + nbcols) * 2, POSITIVE_INFINITY);
            auto* buffers = buffers_v.data();
            size_t c{0+1}, p{nbcols+2};

            // Line & column counters
            size_t i{0}, j{0};

            // Cost accumulator. Also used as the "left neighbour".
            double cost{0};

            // EAP variables: track where to start the next line, and the position of the previous pruning point.
            // Must be init to 0: index 0 is the next starting index and also the "previous pruning point"
            size_t next_start{0}, prev_pp{0};

            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            // Initialisation of the top border: already initialized to +INF. Initialise the left corner to 0.
            buffers[c-1] = 0;

            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            // Main loop
            for (; i < nblines; ++i) {
                // --- --- --- Swap and variables init
                std::swap(c, p);
                const double li = lines[i];
                const size_t jStart = std::max(cap_start_index_to_window(i, w), next_start);
                const size_t jStop = cap_stop_index_to_window_or_end(i, w, nbcols);
                next_start = jStart;
                size_t curr_pp = next_start; // Next pruning point init at the start of the line
                j = next_start;
                // --- --- --- Stage 0: Initialise the left border
                {
                    cost = POSITIVE_INFINITY;
                    buffers[c+jStart-1] = cost;
                }
                // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
                for (; j == next_start && j < prev_pp; ++j) {
                    const auto d = dist(li, cols[j]);
                    cost = std::min(buffers[p + j - 1], buffers[p + j]) + d;
                    buffers[c + j] = cost;
                    if (cost <= ub) { curr_pp = j + 1; } else { ++next_start; }
                }
                // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
                for (; j < prev_pp; ++j) {
                    const auto d = dist(li, cols[j]);
                    cost = min(cost, buffers[p + j - 1], buffers[p + j]) + d;
                    buffers[c + j] = cost;
                    if (cost <= ub) { curr_pp = j + 1; }
                }
                // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
                if (j < jStop) { // If so, two cases.
                    const auto d = dist(li, cols[j]);
                    if (j == next_start) { // Case 1: Advancing next start: only diag.
                        cost = buffers[p + j - 1] + d;
                        buffers[c + j] = cost;
                        if (cost <= ub) { curr_pp = j + 1; }
                        else {
                            // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
                            if (i == nblines - 1 && j == nbcols - 1 && cost <= cutoff) { return cost; }
                            else { return POSITIVE_INFINITY; }
                        }
                    } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
                        cost = std::min(cost, buffers[p + j - 1]) + d;
                        buffers[c + j] = cost;
                        if (cost <= ub) { curr_pp = j + 1; }
                    }
                    ++j;
                } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
                    if (j == next_start) {
                        // But only if we are above the original UB
                        // Else set the next starting point to the last valid column
                        if (cost > cutoff) { return POSITIVE_INFINITY; }
                        else { next_start = nbcols - 1; }
                    }
                }
                // --- --- --- Stage 4: After the previous pruning point: only prev.
                // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
                for (; j == curr_pp && j < jStop; ++j) {
                    const auto d = dist(li, cols[j]);
                    cost = cost + d;
                    buffers[c + j] = cost;
                    if (cost <= ub) { ++curr_pp; }
                }
                // --- --- ---
                prev_pp = curr_pp;
            } // End of main loop for(;i<nblines;++i)

            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            // Finalisation
            // Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
            if (j == nbcols && cost <= cutoff) { return cost; }
            else { return POSITIVE_INFINITY; }
        }

    } // End of namespace internal

    // --- --- --- --- ---
    // --- CDTW
    // --- --- --- --- ---

    /** Constrained Dynamic Time Warping.
     *  Double buffered implementation using O(n) space.
     *  Compute an upper bound before calling the function with pruning.
     *  Any valid path in the cost matrix represent an upper bound.
     *  We compute such a path using the diagonal, then going down in the last column when lengths are disparate.
     * @tparam FloatType    The floating number type used to represent the series.
     * @tparam dist     Distance function, default to square euclidean distance for FloatType.
     * @param series1   Pointer to the first series' values.
     * @param length1   Length of the first series. Must be < tempo::MAX_SERIES_LENGTH.
     * @param series2   Pointer to the second series' values.
     * @param length2   Length of the second series. Must be < tempo::MAX_SERIES_LENGTH.
     * @param w         Half-window parameter (looking at w cells on each side of the diagonal)
     * @return  DTW between the two series
     */
    template<typename FloatType = double, auto dist = square_dist < FloatType>>
    [[nodiscard]] FloatType cdtw(
            const FloatType *series1, size_t length1,
            const FloatType *series2, size_t length2,
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

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Compute a cutoff point using the diagonal
        FloatType cutoff{0};
        // Counter, will first go over the columns, and then complete the lines
        size_t i{0};
        // We have less columns than lines: cover all the columns first.
        for (; i < nbcols; ++i) { cutoff += dist(lines[i], cols[i]); }
        // Then go down in the last column
        if(i<nblines) {
            const auto lc = cols[nbcols - 1];
            for (; i < nblines; ++i) { cutoff += dist(lines[i], lc); }
        }

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        return internal::cdtw<FloatType, dist>(lines, nblines, cols, nbcols, w, cutoff);
    }

    /// Helper for the above, using vectors
    template<typename FloatType=double, auto dist = square_dist < FloatType>>
    [[nodiscard]] inline FloatType cdtw(
            const std::vector<FloatType>& series1,
            const std::vector<FloatType>& series2,
            const size_t w){
        return cdtw<FloatType, dist>(USE(series1), USE(series2), w);
    }

    // --- --- --- --- ---
    // --- DTW with cutoff
    // --- --- --- --- ---

    /** Dynamic Time Warping with cutoff point for early abandoning and pruning.
     *  Double buffered implementation using O(n) space.
     *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning).
     *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
     * @tparam FloatType    The floating number type used to represent the series.
     * @tparam dist     Distance function, default to square euclidean distance for FloatType.
     * @param series1   Pointer to the first series' values.
     * @param length1   Length of the first series. Must be < tempo::MAX_SERIES_LENGTH.
     * @param series2   Pointer to the second series' values.
     * @param length2   Length of the second series. Must be < tempo::MAX_SERIES_LENGTH.
     * @param w         Half-window parameter (looking at w cells on each side of the diagonal)
     * @param cutoff.   Attempt to prune computation of alignments with cost > cutoff.
     *                  May lead to early abandoning.
     * @return  DTW between the two series or +INF if early abandoned.
     *          Warning: a result different from +INF does not warrant a cost < cutoff.
     */
    template<typename FloatType = double, auto dist = square_dist < FloatType>>
    [[nodiscard]] FloatType cdtw(
            const FloatType *series1, size_t length1,
            const FloatType *series2, size_t length2,
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

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        return internal::cdtw<FloatType, dist>(lines, nblines, cols, nbcols, w, cutoff);
    }

    /// Helper for the above, using vectors
    template<typename FloatType=double, auto dist = square_dist<FloatType>>
    [[nodiscard]] inline FloatType cdtw(
            const std::vector<FloatType>& series1,
            const std::vector<FloatType>& series2,
            const size_t w,
            FloatType cutoff){
        return cdtw<FloatType, dist>(USE(series1), USE(series2), w, cutoff);
    }

} // End of namespace tempo::univariate
