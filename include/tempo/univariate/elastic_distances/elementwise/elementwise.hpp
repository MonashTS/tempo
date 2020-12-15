#pragma once

#include "../../../utils/utils.hpp"
#include "../distances.hpp"

namespace tempo::univariate::elastic_distances {

    // --- --- --- --- --- ---
    // Element Wise
    // --- --- --- --- --- ---

    /** Element wise distance. Default to squared euclidean distance.
     * Only defined for same length series (return +INF if different length).
     * @tparam FloatType    The floating number type used to represent the series.
     * @tparam dist     Distance function, default to square euclidean distance
     * @param series1   First series
     * @param length1   Length of the first series
     * @param series2   Second series
     * @param length2   Length of the second series
     * @return Sum of element wise distances or +INF if different lengths
     */
    template<typename FloatType=double, auto dist = square_dist<FloatType>>
    [[nodiscard]] inline double elementwise(
            const FloatType *series1, size_t length1,
            const FloatType *series2, size_t length2
    ) {
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        constexpr auto POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<FloatType>;
        // Pre-conditions. Accept nullptr if length is 0
        assert((series1 != nullptr || length1 == 0) && length1 < MAX_SERIES_LENGTH);
        assert((series2 != nullptr || length2 == 0) && length2 < MAX_SERIES_LENGTH);
        // Check sizes. If both series are empty, return 0, else if one is empty and not the other, maximal error.
        if (length1 != length2) { return POSITIVE_INFINITY; }
        // Compute the Euclidean-like distance
        FloatType cost = 0.0;
        for (size_t i{0}; i < length1; ++i) { cost += dist(series1[i], series2[i]); }
        return cost;
    }

    /// Helper for the above, using vectors
    template<typename FloatType=double, auto dist = square_dist<FloatType>>
    [[nodiscard]] inline FloatType elementwise(
            const std::vector<FloatType> &series1,
            const std::vector<FloatType> &series2) {
        return elementwise<FloatType, dist>(series1.data(), series1.size(), series2.data(), series2.size());
    }


    // --- --- --- --- --- ---
    // Element Wise with cut-off
    // --- --- --- --- --- ---

    /** Element wise distance with cut-off point for early abandoning.
     * Default to squared euclidean distance.
     * Only defined for same length series (return +INF if different length).
     * @tparam FloatType    The floating number type used to represent the series.
     * @tparam dist     Distance function, default to square euclidean distance
     * @param series1   First series
     * @param length1   Length of the first series
     * @param series2   Second series
     * @param length2   Length of the second series
     * @return Sum of element wise distances or +INF if different lengths
     */
    template<typename FloatType=double, auto dist = square_dist<FloatType>>
    [[nodiscard]] inline double elementwise(
            const FloatType *series1, size_t length1,
            const FloatType *series2, size_t length2,
            FloatType cutoff
    ) {
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        constexpr auto POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<FloatType>;
        // Pre-conditions. Accept nullptr if length is 0
        assert((series1 != nullptr || length1 == 0) && length1 < MAX_SERIES_LENGTH);
        assert((series2 != nullptr || length2 == 0) && length2 < MAX_SERIES_LENGTH);
        // Check sizes. If both series are empty, return 0, else if one is empty and not the other, maximal error.
        if (length1 != length2) { return POSITIVE_INFINITY; }
        // Adjust the lower bound, taking the last alignment into account
        const FloatType lastA = dist(series1[length1-1], series2[length1-1]);
        const FloatType original_ub = std::nextafter(cutoff, POSITIVE_INFINITY);
        const FloatType ub = original_ub - lastA;
        // Compute the Euclidean-like distance up to, excluding, the last alignment
        double cost = 0;
        for (size_t i{0}; i < length1-1; ++i) { // Stop before the last: counted in the bound!
            cost += dist(series1[i], series2[i]);
            if(cost>ub){return POSITIVE_INFINITY;}
        }
        // Add the last alignment and check the result
        cost+=lastA;
        if(cost>original_ub){return POSITIVE_INFINITY;} else { return cost; }
    }

    /// Helper for the above, using vectors
    template<typename FloatType=double, auto dist = square_dist<FloatType>>
    [[nodiscard]] inline FloatType elementwise(
            const std::vector<FloatType> &series1,
            const std::vector<FloatType> &series2,
            FloatType cutoff
    ) {
        return elementwise<FloatType, dist>(USE(series1), USE(series2), cutoff);
    }

} // End of namespace tempo::univariate::elastic_distances
