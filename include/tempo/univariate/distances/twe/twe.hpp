#pragma once

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
            assert(w <= nblines);
            assert(nblines-nbcols<=w);
            // Adapt constants to the floating point type
            constexpr auto POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<FloatType>;

            return POSITIVE_INFINITY;
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
                // Compute a cutoff point using the diagonal.
                FloatType cutoff{POSITIVE_INFINITY<double>};
                // We have less columns than lines: cover all the columns first.
                for (size_t i{0}; i < nbcols; ++i) {
                    cutoff += 0; // TODO } // Diag
                    // Then go down in the last column
                    if (nbcols < nblines) {
                        for (size_t i{nbcols}; i < nblines; ++i) {
                            cutoff += 0; // TOP  TODO
                        }
                    }
                }

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
        return twe<FloatType, dist>(USE(series1), USE(series2), nu, lambda);
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
        return twe<FloatType, dist>(USE(series1), USE(series2), nu, lambda, cutoff);
    }

} // End of namespace tempo::univariate
