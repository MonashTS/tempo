#pragma once

#include "../../../utils/utils.hpp"
#include "../distances.hpp"

namespace tempo::univariate {

    namespace internal {

        /** Weighted Dynamic Time Warping with cutoff point for early abandoning and pruning.
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
         * @param weights   Pointer to the weights. Must be at least as long as nblines.
         * @param cutoff.   Attempt to prune computation of alignments with cost > cutoff.
         *                  May lead to early abandoning.
         * @return WTW between the two series or +INF if early abandoned.
         */
        template<typename FloatType = double, auto dist = square_dist < FloatType>>
        [[nodiscard]] inline FloatType wdtw(
                const FloatType *lines, size_t nblines,
                const FloatType *cols, size_t nbcols,
                const FloatType *weights,
                FloatType cutoff
        ) {
            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            // In debug mode, check preconditions
            assert(lines != nullptr && nblines != 0 && nblines < MAX_SERIES_LENGTH);
            assert(cols != nullptr && nbcols != 0 && nbcols < MAX_SERIES_LENGTH);
            assert(nbcols <= nblines);
            // Adapt constants to the floating point type
            constexpr auto POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<FloatType>;

            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            // Create a new tighter upper bounds (most commonly used in the code).
            // First, take the "next float" after "cutoff" to deal with numerical instability.
            // Then, subtract the cost of the last alignment.
            const FloatType ub = initBlock{
                const auto ll = nblines-1;
                const auto lc = nbcols-1;  // Precondition: ll>=lc, so ll-lc>=0, well defined for unsigned size_t.
                return nextafter(cutoff, POSITIVE_INFINITY) -dist(lines[ll], cols[lc])*weights[ll-lc];
            };

            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
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

            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            // Initialisation of the top border: already initialized to +INF. Initialise the left corner to 0.
            buffers[c-1] = 0;

            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            // Main loop
            for (; i < nblines; ++i) {
                // --- --- --- Swap and variables init
                std::swap(c, p);
                const double li = lines[i];
                size_t curr_pp = next_start; // Next pruning point init at the start of the line
                j = next_start;
                // --- --- --- Stage 0: Initialise the left border
                {
                    cost = POSITIVE_INFINITY;
                    buffers[c+next_start-1] = POSITIVE_INFINITY;
                }
                // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
                for (; j == next_start && j < prev_pp; ++j) {
                    const auto d = dist(li, cols[j])*weights[absdiff(i,j)];
                    cost = std::min(buffers[p + j - 1], buffers[p + j]) + d;
                    buffers[c + j] = cost;
                    if (cost <= ub) { curr_pp = j + 1; } else { ++next_start; }
                }
                // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
                for (; j < prev_pp; ++j) {
                    const auto d = dist(li, cols[j])*weights[absdiff(i,j)];
                    cost = min(cost, buffers[p + j - 1], buffers[p + j]) + d;
                    buffers[c + j] = cost;
                    if (cost <= ub) { curr_pp = j + 1; }
                }
                // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
                if (j < nbcols) { // If so, two cases.
                    const auto d = dist(li, cols[j])*weights[absdiff(i,j)];
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
                for (; j == curr_pp && j < nbcols; ++j) {
                    const auto d = dist(li, cols[j])*weights[absdiff(i,j)];
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
    // Weights generation
    // --- --- --- --- ---

    /// From the paper, changing this values does not change the results (scaling), so keep to 1
    constexpr double WDTW_MAX_WEIGHT = 1;

    /** Compute a weight at index i in a sequence 1..m
     * @param g "Controls the level of penalization for the points with larger phase difference".
     *        range [0, +inf), usually in [0.01, 0.6].
     *        Some examples:
     *        * 0: constant weight
     *        * 0.05: nearly linear weights
     *        * 0.25: sigmoid weights
     *        * 3: two distinct weights between half sequences
     *
     * @param half_max_length Mid point of the sequence (m/2)
     * @param i Index of the point in [1..m] (m=length of the sequence)
     * @return the weight for index i
     */
    template<typename FloatType=double>
    [[nodiscard]] inline FloatType compute_weight(FloatType g, FloatType half_max_length, FloatType i) {
        return WDTW_MAX_WEIGHT / (1 + exp(-g * (i - half_max_length)));
    }

    /// Populate the weights_array of size length with weights derive from the g factor
    template<typename FloatType=double>
    inline void populate_weights(FloatType g, FloatType *weights_array, size_t length) {
        FloatType half_max_length = FloatType(length) / 2;
        for (size_t i{0}; i < length; ++i) {
            weights_array[i] = compute_weight(g, half_max_length, FloatType(i));
        }
    }

    /// Create a vector of weights
    template<typename FloatType=double>
    inline std::vector<FloatType> generate_weights(FloatType g, size_t length){
        std::vector<FloatType> weights(length, 0);
        populate_weights(g, weights.data(), length);
        return weights;
    }


    // --- --- --- --- ---
    // WDTW
    // --- --- --- --- ---

   /** Weighted Dynamic Time Warping.
    *  Double buffered implementation using O(n) space.
    *  Compute an upper bound before calling the function with pruning.
    *  Any valid path in the cost matrix represent an upper bound. We compute such a path using the squared euclidean
    *  distance, augmented by a straight line when length are disparate.
    * @tparam FloatType    The floating number type used to represent the series.
    * @tparam dist     Distance function, default to square euclidean distance for FloatType.
    * @param series1   Pointer to the first series' values.
    * @param length1   Length of the first series. Must be < tempo::MAX_SERIES_LENGTH.
    * @param series2   Pointer to the second series' values.
    * @param length2   Length of the second series. Must be < tempo::MAX_SERIES_LENGTH.
    * @param weights   Pointer to the weights. Must be at least as long as nblines.
    * @return WTW between the two series or +INF if early abandoned.
    */
    template<typename FloatType = double, auto dist = square_dist < FloatType>>
    [[nodiscard]] FloatType wdtw(
            const FloatType *series1, size_t length1,
            const FloatType *series2, size_t length2,
            const FloatType *weights
    ) {
       const auto check_result = check_order_series(series1, length1, series2, length2);
       switch (check_result.index()) {
           case 0: { return std::get<0>(check_result);}
           case 1: {
               const auto[lines, nblines, cols, nbcols] = std::get<1>(check_result);

               // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
               // Compute a cutoff point using the diagonal
               FloatType cutoff{0};
               // We have less columns than lines: cover all the columns first.
               for (size_t i{0}; i < nbcols; ++i) { cutoff += dist(lines[i], cols[i])*weights[0]; }
               // Then go down in the last column
               if(nbcols<nblines) {
                   const auto lc = cols[nbcols - 1];
                   for (size_t i{nbcols}; i < nblines; ++i) { cutoff += dist(lines[i], lc)*weights[i-nbcols+1]; }
               }

               // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
               return internal::wdtw<FloatType, dist>(lines, nblines, cols, nbcols, weights, cutoff);
           }
           default: should_not_happen();
       }
    }

    /// Helper for the above, using vectors
    template<typename FloatType=double, auto dist = square_dist < FloatType>>
    [[nodiscard]] inline FloatType wdtw(
            const std::vector<FloatType>& series1,
            const std::vector<FloatType>& series2,
            const std::vector<FloatType>& weights
            ){
        assert(weights.size() >= std::max(series1.size(), series2.size()));
        return wdtw<FloatType, dist>(series1.data(), series1.size(), series2.data(), series2.size(), weights.data());
    }

    // --- --- --- --- ---
    // --- WDTW with cutoff
    // --- --- --- --- ---

    /** Weighted Dynamic Time Warping with cutoff point for early abandoning and pruning.
     *  Double buffered implementation using O(n) space.
     *  Worst case scenario has a O(n²) time complexity (no pruning nor early abandoning).
     *  A tight cutoff can allow a lot of pruning, speeding up the process considerably.
     * @tparam FloatType    The floating number type used to represent the series.
     * @tparam dist     Distance function, default to square euclidean distance for FloatType.
     * @param series1   Pointer to the first series' values.
     * @param length1   Length of the first series. Must be < tempo::MAX_SERIES_LENGTH.
     * @param series2   Pointer to the second series' values.
     * @param length2   Length of the second series. Must be < tempo::MAX_SERIES_LENGTH.
     * @param weights   Pointer to the weights. Must be at least as long as nblines.
     * @param cutoff.   Attempt to prune computation of alignments with cost > cutoff.
     *                  May lead to early abandoning.
     * @return WTW between the two series or +INF if early abandoned.
     */
    template<typename FloatType = double, auto dist = square_dist < FloatType>>
    [[nodiscard]] FloatType wdtw(
            const FloatType *series1, size_t length1,
            const FloatType *series2, size_t length2,
            const FloatType *weights,
            FloatType cutoff
    ) {
        const auto check_result = check_order_series(series1, length1, series2, length2);
        switch (check_result.index()) {
            case 0: { return std::get<0>(check_result);}
            case 1: {
                const auto[lines, nblines, cols, nbcols] = std::get<1>(check_result);
                return internal::wdtw<FloatType, dist>(lines, nblines, cols, nbcols, weights, cutoff);
            }
            default: should_not_happen();
        }
    }

    /// Helper for the above, using vectors
    template<typename FloatType=double, auto dist = square_dist<FloatType>>
    [[nodiscard]] inline FloatType wdtw(
            const std::vector<FloatType>& series1,
            const std::vector<FloatType>& series2,
            const std::vector<FloatType>& weights,
            FloatType cutoff){
        assert(weights.size() >= std::max(series1.size(), series2.size()));
        return wdtw<FloatType, dist>(series1.data(), series1.size(), series2.data(), series2.size(), weights.data(), cutoff);
    }


} // End of namespace tempo::univariate
