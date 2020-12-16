#pragma once

#include <variant>
#include <tuple>

namespace tempo::univariate {

    /// Helper macro converting a "vector" into "pointer, length"
    #define USE(series) series.data(), series.size()

    /// Type alias for a tuple representing (lines, nblines, cols, nbcosl)
    template<typename FloatType=double>
    using lico_t = std::tuple<const FloatType*, size_t, const FloatType*, size_t>;

    /// Helper function checking and ordering the length of the series
    template<typename FloatType=double>
    [[nodiscard]] inline std::variant<FloatType, lico_t<FloatType>> check_order_series(
            const FloatType *series1, size_t length1,
            const FloatType *series2, size_t length2
            ){
        constexpr auto POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<FloatType>;
        // Pre-conditions. Accept nullptr if length is 0
        assert((series1 != nullptr || length1 == 0) && length1 < MAX_SERIES_LENGTH);
        assert((series2 != nullptr || length2 == 0) && length2 < MAX_SERIES_LENGTH);
        // Check sizes. If both series are empty, return 0, else if one is empty and not the other, maximal error.
        if (length1 == 0 && length2 == 0) { return {FloatType(0.0)}; }
        else if ((length1 == 0) != (length2 == 0)) { return POSITIVE_INFINITY; }
        // Use the smallest size as the columns (which will be the allocation size)
        return (length1 > length2) ?
            std::tuple(series1, length1, series2, length2) :
            std::tuple(series2, length2, series1, length1);
    }

    /// Square distances between two numeric values
    template<typename T>
    [[nodiscard]] inline T square_dist(T a, T b){
        const auto d = a-b;
        return d*d;
    }

}
