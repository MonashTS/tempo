#pragma once

#include <tempo/univariate/distances/dtw/dtw.hpp>
#include <tempo/univariate/distances/dtw/cdtw.hpp>
#include <tempo/univariate/distances/dtw/wdtw.hpp>
namespace cpp = tempo::univariate;

#include "../../utils.hpp"

namespace pytempo::univariate {

    // --- --- --- --- --- ---
    // DTW
    // --- --- --- --- --- ---

    inline double dtw(nparray series1, nparray series2) {
        check(series1, series2);
        return cpp::dtw(USE(series1), USE(series2));
    }

    inline double dtw_ea(nparray series1, nparray series2, double cutoff) {
        check(series1, series2);
        return cpp::dtw(USE(series1), USE(series2), cutoff);
    }



    // --- --- --- --- --- ---
    // CDTW
    // --- --- --- --- --- ---

    inline double cdtw(nparray series1, nparray series2, size_t w) {
        check(series1, series2);
        return cpp::cdtw(USE(series1), USE(series2), w);
    }

    inline double cdtw_ea(nparray series1, nparray series2, size_t w, double cutoff) {
        check(series1, series2);
        return cpp::cdtw(USE(series1), USE(series2), w, cutoff);
    }



    // --- --- --- --- --- ---
    // WDTW
    // --- --- --- --- --- ---

    inline double wdtw(nparray series1, nparray series2, nparray weights) {
        check(series1, series2);
        if (weights.size() < std::max(series1.size(), series2.size())) {
            throw std::invalid_argument("Weights array is too short: must be at least as long as the longest series.");
        }
        return cpp::wdtw(USE(series1), USE(series2), weights.data());
    }

    inline double wdtw_ea(nparray series1, nparray series2, nparray weights, double cutoff) {
        check(series1, series2);
        if (weights.size() < std::max(series1.size(), series2.size())) {
            throw std::invalid_argument("Weights array is too short: must be at least as long as the longest series.");
        }
        return cpp::wdtw(USE(series1), USE(series2), weights.data(), cutoff);
    }

    py::array_t<double> wdtw_weights(size_t length, double g) {
        auto result = py::array_t<double>(length);
        cpp::populate_weights(g, result.mutable_data(), result.size());
        return result;
    }



    // --- --- --- --- --- ---
    // Python module
    // --- --- --- --- --- ---

    inline void init_dtw(py::module &m) {
        m.def("dtw", &dtw,
              "DTW between two series.",
              "serie1"_a, "serie2"_a
        );

        m.def("dtw", &dtw_ea,
              "DTW between two series. With pruning & early abandoning cut-off.",
              "serie1"_a, "serie2"_a, "cutoff"_a
        );



        m.def("cdtw", &cdtw,
              "Constrained DTW between two series. \"w\" is the half-window size.",
              "serie1"_a, "serie2"_a, "w"_a
        );

        m.def("cdtw", &cdtw_ea,
              "Constrained DTW between two series. \"w\" is the half-window size. With pruning & early abandoning cut-off.",
              "serie1"_a, "serie2"_a, "w"_a, "cutoff"_a
        );



        m.def("wdtw", &wdtw,
              "Weighted DTW between two series, using an array of weights.",
              "serie1"_a, "serie2"_a, "weights"_a
        );

        m.def("wdtw", &wdtw_ea,
              "Weighted DTW between two series, using an array of weights. With pruning & early abandoning cut-off.",
              "serie1"_a, "serie2"_a, "weights"_a, "cutoff"_a
        );



        m.def("wdtw_weights", &wdtw_weights,
              "Generate a numpy array of weights suitable for WDTW.",
              "length"_a, "g"_a
        );
    }
}