#pragma once

#include <algorithm>
#include <cmath>
#include <tempo/tseries/tseries.hpp>
#include <utility>

#include "../utils.hpp"

namespace pytempo {

    using pyTSeries = tempo::TSeries<double, std::string>;
    namespace details {
        using ft = pyTSeries::FloatType_t;
        constexpr ssize_t scalar_size = sizeof(ft);
        const std::string format = py::format_descriptor<ft>::format();  // Python struct-style format descriptor

        pyTSeries make_pyTseries(nparray a, bool has_missing, std::optional<std::string> label){
            if(a.ndim() == 1){
                return pyTSeries(a.data(), a.shape(0), 1, has_missing, label, std::make_shared<std::any>(a));
            } else if(a.ndim() == 2){
                return pyTSeries(a.data(), a.shape(1), a.shape(0), has_missing, label, std::make_shared<std::any>(a));
            } else {
                throw std::runtime_error("Incompatible buffer dimension (must be 1 or 2).");
            }
        }
    } // End of namespace details

    /// Register the TSeries class
    inline void submod_tseries(py::module& m){

        // --- --- --- TSeries
        py::class_<pyTSeries, std::shared_ptr<pyTSeries>>(m, "TSeries", py::buffer_protocol())
                .def("length", &pyTSeries::length, "Length of the time series")
                .def("ndim", &pyTSeries::nb_dimensions, "Number of 'dimensions' of the series (i.e. number of 'tracks')")
                .def("has_missing", &pyTSeries::has_missing, "Has missing values?")
                .def("label", &pyTSeries::label, "Get the label (a string)")
                        /// Bare bones interface
                .def("__getitem__", [](const pyTSeries &s, std::pair<py::ssize_t, py::ssize_t> i) {
                    if (i.first >= (ssize_t)s.nb_dimensions()
                        || i.second >= (ssize_t)s.length()
                        || i.first <0 || i.second<0){
                        throw py::index_error();
                    }
                    return s(i.first, i.second);
                })
                .def("__getitem__", [](const pyTSeries &s, py::ssize_t i) {
                    if(s.nb_dimensions()>1){
                        throw std::invalid_argument("A multivariate series must be accessed by [track, idx]");
                    }
                    if (i<0 || i>=(ssize_t)s.length()){
                        throw py::index_error();
                    }
                    return s.data()[i];
                })
                // Construct an empty default TSeries
                .def(py::init())
                // Construct from numpy array
                .def(py::init(&details::make_pyTseries),
                     "Wrap an array in a TSeries object. The backing array must not be modified.",
                     "array"_a, "has_missing"_a, "label"_a
                )
                .def(py::init([](nparray a, std::optional<std::string> label){
                     bool missing = std::any_of(a.data(), a.data()+a.size(), (bool(*)(double))std::isnan);
                     return details::make_pyTseries(a, missing, std::move(label));
                    }),
                 "Wrap an array in a TSeries object. The backing array must not be modified."
                 "O(n): check the series for missing values",
                 "array"_a, "label"_a
                )
                // To Buffer protocol
                .def_buffer([](const pyTSeries& series)->py::buffer_info{
                    using namespace details;
                    ssize_t l = series.length();                    // convert to ssize_t
                    ssize_t nb_tracks = series.nb_dimensions();     // convert to ssize_t
                    // Number of dimensions: multivariate if more than 1 track
                    if(nb_tracks==1){ // Univariate

                        return py::buffer_info {
                                (void*)series.data(),           // Pointer to buffer
                                scalar_size,                    // Size of one scalar
                                format,                         // Python struct-style format descriptor
                                1,                              // Number of dimensions
                                {l},                            // Shape = length of the series
                                {scalar_size},                  // Strides (in bytes) for each index
                                true                            // Readonly = true
                        };

                    } else { // Multivariate

                        return py::buffer_info {
                                (void*)series.data(),           // Pointer to buffer
                                scalar_size,                    // Size of one scalar
                                format,                         // Python struct-style format descriptor
                                2,                              // Number of dimensions
                                {nb_tracks, l},                 // "First dimension" is the number of track, then the length of the tracks
                                {scalar_size*l, scalar_size},   // Strides (in bytes) for each index
                                true                            // Readonly = true
                        };

                    }

                });
    }

}
