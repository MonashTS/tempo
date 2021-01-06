#pragma once

#include <tempo/tseries/tseries.hpp>

#include "../utils.hpp"

namespace pytempo {

    using pyTSeries = tempo::TSeries<double, std::string>;
    namespace details {
        using ft = pyTSeries::FloatType_t;
        constexpr ssize_t scalar_size = sizeof(ft);
        const std::string format = py::format_descriptor<ft>::format();  // Python struct-style format descriptor

    } // End of namespace details

    /// Register the TSeries class
    inline void submod_tseries(py::module& m){

        // --- --- --- TSeries
        py::class_<pyTSeries, std::shared_ptr<pyTSeries>>(m, "TSeries", py::buffer_protocol())
        //py::class_<pyTSeries>(m, "TSeries") // , py::buffer_protocol())
        .def("length", &pyTSeries::length)
        .def(py::init())
        // Construct from numpy array
        .def(py::init([](nparray a, bool has_missing, std::optional<std::string> label){
            if(a.ndim() == 1){
                return pyTSeries(a.data(), a.shape(0), 1, has_missing, label);
            } else if(a.ndim() == 2){
                return pyTSeries(a.data(), a.shape(1), a.shape(0), has_missing, label);
            } else {
                throw std::runtime_error("Incompatible buffer dimension (must be 1 or 2).");
            }
        }))
        // To Buffer protocol
        .def_buffer([](const pyTSeries& series)->py::buffer_info{
            using namespace details;
            ssize_t l = series.length();            // convert to ssize_t
            ssize_t d = series.nb_dimensions();     // convert to ssize_t
            return py::buffer_info {
                    (void*)series.data(),           // Pointer to buffer
                    scalar_size,                    // Size of one scalar
                    format,                         // Python struct-style format descriptor
                    2,                              // Number of dimensions
                    {d, l},                         // "First dimension" is the number of track, then the length of the tracks
                    {scalar_size*l, scalar_size},   // Strides (in bytes) for each index
                    true                            // Readonly = true
            };
        });
    }

}
