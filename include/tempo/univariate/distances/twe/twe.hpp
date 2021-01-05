#pragma once

#include "../../../tseries/tseries.hpp"
#include "../../../utils/utils.hpp"
#include "../distances.hpp"

namespace tempo::univariate {

    namespace internal {

        /** Time Warp Edit distance with cutoff point for early abandoning and pruning.
         *  Double buffered implementation using O(n) space.
         *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning).
         *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
         *  Actual implementation assuming that some pre-conditions are fulfilled.
         * @tparam FloatType    The floating number type used to represent the series.
         * @tparam dist     Distance function, default to square euclidean distance for FloatType.
         * @param lines     Pointer to the "line series". Must be the longest series. Cannot be null.
         * @param nblines   Length of the line series. Must be 0 < nbcols <= nblines < tempo::MAX_SERIES_LENGTH.
         * @param cols      Pointer to the "column series". Must be the shortest series. Cannot be null.
         * @param nbcols    Length of the column series. Must be 0 < nbcols <= nblines < tempo::MAX_SERIES_LENGTH.
         * @param nu        Stiffness parameter
         * @param lambda    Penalty parameter
         * @param cutoff.   Attempt to prune computation of alignments with cost > cutoff.
         *                  May lead to early abandoning.
         * @return TWE cost or +INF if early abandoned
         */
        template<typename FloatType = double, auto dist = square_dist < FloatType>>
        [[nodiscard]] FloatType twe(
                const FloatType *lines, size_t nblines,
                const FloatType *cols, size_t nbcols,
                const FloatType nu, const FloatType lambda,
                const FloatType cutoff
        ) {
            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            // In debug mode, check preconditions
            assert(lines != nullptr && nblines != 0 && nblines < MAX_SERIES_LENGTH);
            assert(cols != nullptr && nbcols != 0 && nbcols < MAX_SERIES_LENGTH);
            assert(nbcols <= nblines);
            // Adapt constants to the floating point type
            constexpr auto POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<FloatType>;

            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            // Constants: we only consider timestamp spaced by 1, so:
            // In the "delete" case, we always have a time difference of 1, so we always have 1*nu+lambda
            const auto nu_lambda = nu+lambda;
            // In the "match" case, we always have nu*(|i-j|+|(i-1)-(j-1)|) == 2*nu*|i-j|
            const auto nu2 = FloatType(2)*nu;

            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            // Create a new tighter upper bounds (most commonly used in the code).
            // First, take the "next float" after "cutoff" to deal with numerical instability.
            // Then, subtract the cost of the last alignment.
            const FloatType ub = initBlock {
                // The last alignment can only computed if we have nbcols >= 2
                if(nbcols>=2) {
                    const auto li = lines[nblines - 1];
                    const auto li1 = lines[nblines - 2];
                    const auto co = cols[nbcols - 1];
                    const auto co1 = cols[nbcols - 2];
                    const double distli = dist(li1, li);
                    const double distco = dist(co1, co);
                    const auto la = min(
                            distco + nu_lambda,                                     // "Delete_B": over the columns / Prev
                            dist(li, co) + dist(li1, co1) + nu2 * (nblines-nbcols), // Match: Diag. Ok: nblines >= nbcols
                            distli + nu_lambda                                      // "Delete_A": over the lines / Top
                    );
                    return FloatType(nextafter(cutoff, POSITIVE_INFINITY) - la);
                } else {
                    return FloatType(cutoff);
                }
            };

            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            // Double buffer allocation, no initialisation required (border condition manage in the code).
            // Base indices for the 'c'urrent row and the 'p'revious row.
            auto buffers = std::unique_ptr<FloatType[]>(new FloatType[nbcols * 2]);
            size_t c{0}, p{nbcols};

            // Buffer holding precomputed distance between columns
            auto distcol = std::unique_ptr<FloatType[]>(new FloatType[nbcols]);

            // Line & column counters
            size_t i{0}, j{0};

            // Cost accumulator. Also used as the "left neighbour".
            FloatType cost;

            // EAP variables: track where to start the next line, and the position of the previous pruning point.
            // Must be init to 0: index 0 is the next starting index and also the "previous pruning point"
            size_t next_start{0}, prev_pp{0};

            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            // Initialisation of the first line. Deal with the line top border condition.
            {
                // Case [0,0]: special "Match case"
                cost = dist(lines[0], cols[0]);
                buffers[c + 0] = cost;
                // Distance for the first column is relative to 0 "by conventions" (from the paper, section 4.2)
                distcol[0]=dist(0, cols[0]);
                // Rest of the line: [i==0, j>=1]: "Delete_B case" (prev)
                // We also initialize 'distcol' here.
                for (j = 1; j < nbcols; ++j) {
                    const double d = dist(cols[j - 1], cols[j]);
                    distcol[j] = d;
                    cost = cost + d + nu_lambda;
                    buffers[c + j] = cost;
                    if (cost <= ub) { prev_pp = j + 1; } else {break;}
                }
                // Complete the initialisation of distcol
                for (; j < nbcols; ++j) {
                    const double d = dist(cols[j - 1], cols[j]);
                    distcol[j] = d;
                }
                // Next line.
                ++i;
            }


            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            // Main loop, starts at the second line
            for (; i < nblines; ++i) {
                // --- --- --- Swap and variables init
                std::swap(c, p);
                const double li = lines[i];
                const double li1 = lines[i - 1];
                const double distli = dist(li1, li);
                size_t curr_pp = next_start; // Next pruning point init at the start of the line
                j = next_start;
                // --- --- --- Stage 0: Special case for the first column. Can only look up (border on the left)
                {
                    cost = buffers[p + j] + distli + nu_lambda; // "Delete_A" / Top
                    buffers[c + j] = cost;
                    if (cost <= ub) { curr_pp = j + 1; } else { ++next_start; }
                    ++j;
                }
                // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
                for (; j == next_start && j < prev_pp; ++j) {
                    cost = std::min(
                            buffers[p + j - 1] + dist(li, cols[j]) + dist(li1, cols[j - 1]) + nu2 * absdiff(i, j), // "Match" / Diag
                            buffers[p + j] + distli + nu_lambda // "Delete_A" / Top
                    );
                    buffers[c + j] = cost;
                    if (cost <= ub) { curr_pp = j + 1; } else { ++next_start; }
                }
                // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
                for (; j < prev_pp; ++j) {
                    cost = min(
                            cost + distcol[j] + nu_lambda,      // "Delete_B": over the columns / Prev
                            buffers[p + j - 1] + dist(li, cols[j]) + dist(li1, cols[j - 1]) + nu2 * absdiff(i, j), // Match: Diag
                            buffers[p + j] + distli + nu_lambda // "Delete_A": over the lines / Top
                    );
                    buffers[c + j] = cost;
                    if (cost <= ub) { curr_pp = j + 1; }
                }
                // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
                if (j < nbcols) { // If so, two cases.
                    if (j == next_start) { // Case 1: Advancing next start: only diag.
                        cost = buffers[p + j - 1] + dist(li, cols[j]) + dist(li1, cols[j - 1]) + nu2 * absdiff(i, j); // Match: Diag
                        buffers[c + j] = cost;
                        if (cost <= ub) { curr_pp = j + 1; }
                        else {
                            // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
                            if (i == nblines - 1 && j == nbcols - 1 && cost <= cutoff) { return cost; }
                            else { return POSITIVE_INFINITY; }
                        }
                    } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
                        cost = std::min(
                                cost + distcol[j] + nu_lambda,      // "Delete_B": over the columns / Prev
                                buffers[p + j - 1] + dist(li, cols[j]) + dist(li1, cols[j - 1]) + nu2 * absdiff(i, j) // Match: Diag
                        );
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
                for (; j == curr_pp && j < nbcols; ++j) {
                    cost = cost + distcol[j] + nu_lambda; // "Delete_B": over the columns / Prev
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
    // --- TWE
    // --- --- --- --- ---

    /** Time Warp Edit distance.
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
     * @param nu        Stiffness parameter
     * @param lambda    Penalty parameter
     * @return TWE cost between the two series
     */
    template<typename FloatType = double, auto dist = square_dist < FloatType>>
    [[nodiscard]] FloatType twe(
            const FloatType *series1, size_t length1,
            const FloatType *series2, size_t length2,
            const FloatType nu, const FloatType lambda
    ) {
        const auto check_result = check_order_series(series1, length1, series2, length2);
        switch (check_result.index()) {
            case 0: { return std::get<0>(check_result); }
            case 1: {
                const auto[lines, nblines, cols, nbcols] = std::get<1>(check_result);

                // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                // Compute a cutoff point using the diagonal. Init with the first cell at (0,0)
                FloatType cutoff{dist(lines[0], cols[0])};

                // We have less columns than lines: cover all the columns first. Starts at (1,1)
                // Match: diagonal with absdiff(i,i) == 0
                for (size_t i{1}; i < nbcols; ++i) {
                    cutoff = cutoff + dist(lines[i], cols[i]) + dist(lines[i-1], cols[i - 1]);
                }

                // Then go down in the last column
                if (nbcols < nblines) {
                    const auto nu_lambda = nu+lambda;
                    for (size_t i{nbcols}; i < nblines; ++i) {
                        const double distli = dist(lines[i-1], lines[i]);
                        cutoff = cutoff + distli + nu_lambda;
                    }
                }

                // A bit of wiggle room due to floats rounding
                cutoff = nextafter(cutoff, POSITIVE_INFINITY<FloatType>);

                // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                return internal::twe<FloatType, dist>(lines, nblines, cols, nbcols, nu, lambda, cutoff);
            }
            default: should_not_happen();
        }
    }

    /// Helper for the above, using vectors
    template<typename FloatType = double, auto dist = square_dist < FloatType>>
    [[nodiscard]] inline FloatType twe(
            const std::vector<FloatType>& series1,
            const std::vector<FloatType>& series2,
            const FloatType nu, const FloatType lambda
    ){
        return twe<FloatType, dist>(series1.data(), series1.size(), series2.data(), series2.size(), nu, lambda);
    }


    /// Helper for the above, using TSeries
    template<typename FloatType = double, auto dist = square_dist < FloatType>>
    [[nodiscard]] inline FloatType twe(
            const TSeries<FloatType>& series1,
            const TSeries<FloatType>& series2,
            const FloatType nu, const FloatType lambda
    ){
        return twe<FloatType, dist>(series1.data(), series1.length(), series2.data(), series2.length(), nu, lambda);
    }

    // --- --- --- --- ---
    // --- TWE with cutoff
    // --- --- --- --- ---

    /** Time Warp Edit distance with cutoff point for early abandoning and pruning.
     *  Double buffered implementation using O(n) space.
     *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning).
     *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
     *  Actual implementation assuming that some pre-conditions are fulfilled.
     * @tparam FloatType    The floating number type used to represent the series.
     * @tparam dist     Distance function, default to square euclidean distance for FloatType.
     * @param series1   Pointer to the first series' values.
     * @param length1   Length of the first series. Must be < tempo::MAX_SERIES_LENGTH.
     * @param series2   Pointer to the second series' values.
     * @param length2   Length of the second series. Must be < tempo::MAX_SERIES_LENGTH.
     * @param nu        Stiffness parameter
     * @param lambda    Penalty parameter
     * @param cutoff.   Attempt to prune computation of alignments with cost > cutoff.
     *                  May lead to early abandoning.
     * @return TWE cost or +INF if early abandoned
     */
    template<typename FloatType = double, auto dist = square_dist < FloatType>>
    [[nodiscard]] FloatType twe(
            const FloatType *series1, size_t length1,
            const FloatType *series2, size_t length2,
            const FloatType nu, const FloatType lambda,
            const FloatType cutoff
    ) {
        const auto check_result = check_order_series(series1, length1, series2, length2);
        switch (check_result.index()) {
            case 0: { return std::get<0>(check_result);}
            case 1: {
                const auto[lines, nblines, cols, nbcols] = std::get<1>(check_result);
                return internal::twe<FloatType, dist>(lines, nblines, cols, nbcols, nu, lambda, cutoff);
            }
            default: should_not_happen();
        }
    }

    /// Helper for the above, using vectors
    template<typename FloatType = double, auto dist = square_dist < FloatType>>
    [[nodiscard]] inline FloatType twe(
            const std::vector<FloatType>& series1,
            const std::vector<FloatType>& series2,
            const FloatType nu, const FloatType lambda,
            const FloatType cutoff){
        return twe<FloatType, dist>(series1.data(), series1.size(), series2.data(), series2.size(), nu, lambda, cutoff);
    }


    /// Helper for the above, using TSeries
    template<typename FloatType = double, auto dist = square_dist < FloatType>>
    [[nodiscard]] inline FloatType twe(
            const TSeries<FloatType>& series1,
            const TSeries<FloatType>& series2,
            const FloatType nu, const FloatType lambda,
            const FloatType cutoff){
        return twe<FloatType, dist>(series1.data(), series1.length(), series2.data(), series2.length(), nu, lambda, cutoff);
    }

} // End of namespace tempo::univariate
