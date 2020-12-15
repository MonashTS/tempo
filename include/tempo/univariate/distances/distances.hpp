#pragma once


namespace tempo::univariate::distances {

    /// Helper macro
    #define USE(series) series.data(), series.size()

    /// Square distances between two numeric values
    template<typename T>
    [[nodiscard]] inline T square_dist(T a, T b){
        const auto d = a-b;
        return d*d;
    }

}
