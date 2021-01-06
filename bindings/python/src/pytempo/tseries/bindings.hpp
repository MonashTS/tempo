#pragma once

#include <tempo/tseries/tseries.hpp>

#include "../utils.hpp"

namespace pytempo {

    using pyTSeries = tempo::TSeries<double, std::string>;

    /// Register the TSeries class
    inline void submod_tseries(py::module& m){

        // --- --- --- TSeries
        py::class_<pyTSeries, std::shared_ptr<pyTSeries>>(m, "TSeries", py::buffer_protocol())
        .def(py::init())
        .def("length", &pyTSeries::length)
        // Buffer protocol
        .def_buffer([](pyTSeries series)->py::buffer_info{
            using ft = pyTSeries::FloatType_t;
            constexpr ssize_t sft = sizeof(ft);
            ssize_t l = series.length();           // convert to ssize_t
            ssize_t d = series.nb_dimensions();    // convert to ssize_t
            py::buffer_info binfo {
                    (void*)series.data(),                  // Pointer to buffer
                    sft,                                    // Size of one scalar
                    py::format_descriptor<ft>::format(),    // Python struct-style format descriptor
                    2,                                      // Number of dimensions
                    {d, l},                                 // "First dimension" is the number of track, then the length of the tracks
                    {sft*l, sft},                           // Strides (in bytes) for each index
                    true                                    // Readonly = true
            };
            return binfo;
        });
    }

}
