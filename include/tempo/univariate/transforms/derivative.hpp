#pragma once

#include <cstddef>

namespace tempo::univariate {

    /** Computation of a series derivative according to "Derivative Dynamic Time Warping" by Keogh & Pazzani
     * @tparam FloatType    The floating number type used to represent the series.
     * @param series        Pointer to the series's data
     * @param length        Length of the series
     * @param out           Pointer where to write the derivative. Must be able to store 'length' values.
     */
    template<typename FloatType=double>
    void derivative(const FloatType *series, size_t length, FloatType *out) {
        if (length > 2) {
            for (size_t i{1}; i < length - 1; ++i) {
                out[i] = ((series[i] - series[i - 1]) + ((series[i + 1] - series[i - 1]) / 2.0)) / 2.0;
            }
            out[0] = out[1];
            out[length - 1] = out[length - 2];
        } else {
            std::copy(series, series + length, out);
        }
    }

} // End of namespace tempo::univariate
