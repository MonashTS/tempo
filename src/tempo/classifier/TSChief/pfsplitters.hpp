#pragma once

#include <vector>
#include <string>
#include <memory>

#include "tempo/dataset/dts.hpp"
#include "tempo/classifier/TSChief/splitter_interface.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1dist_interface.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1splitter.hpp"

#include "tempo/classifier/TSChief/sleaf/pure_leaf.hpp"

namespace pf::splitters {

    using F = double;
    namespace tsc = tempo::classifier::TSChief;
    namespace tsc_nn1 = tempo::classifier::TSChief::snode::nn1splitter;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Parameterization
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    // --- --- --- Cost function exponent

    /// Give a set of exponents to uniformly choose from
    tsc_nn1::ExponentGetter make_get_vcfe(std::vector<F> exponent_set);

    /// Always return exponent 1.0
    tsc_nn1::ExponentGetter make_get_cfe1();

    /// Always return exponent 2.0
    tsc_nn1::ExponentGetter make_get_cfe2();

    // --- --- --- Transform getters

    /// Give a set of transform names to uniformly choose from
    tsc_nn1::TransformGetter make_get_transform(std::vector<std::string> tr_set);

    /// Get the transform "default"
    tsc_nn1::TransformGetter make_get_default();

    /// Get the transform "derivative<d>" where "<d>" is the degree, e.g. "derivative1"
    tsc_nn1::TransformGetter make_get_derivative(size_t d);

    // --- --- --- Window getter

    /// Given the maximum length of a series, produce a window
    tsc_nn1::WindowGetter make_get_window(size_t maxlength);

    // --- --- --- ERP Gap Value *AND* LCSS epsilon.

    /// Random fraction of the incoming data standard deviation, within [stddev/5, stddev[
    /// Must be able to access the dataset to compute the stddev at the node
    tsc_nn1::StatGetter make_get_frac_stddev();

    // --- --- --- MSM Cost

    tsc_nn1::T_GetterState<F> make_get_msm_cost();

    // --- --- --- TWE nu & lambda parameters

    tsc_nn1::T_GetterState<F> make_get_twe_nu();

    tsc_nn1::T_GetterState<F> make_get_twe_lambda();

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Splitters
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    // --- --- --- Leaf Generator

    std::shared_ptr<tsc::i_GenLeaf> make_pure_leaf(tempo::DatasetHeader const &train_header);

    std::shared_ptr<tsc::i_GenLeaf> make_pure_leaf_smoothp(tempo::DatasetHeader const &train_header);

    /** Generate node splitters for PF (distance splitters)
     * @param exponents           List of exponents for the DTW (including DA) family (uniform choice)
     * @param transforms          List of transforms, for all distances (uniform choice)
     * @param distances           List of distance name (DA, ADTW, DTW, DTWFull, WDTW, ERP, LCSS, MSM, TWE)
     * @param nbc                 Number of distance candidates per node
     * @param series_max_length   Maximum length of the series
     * @param train_data          Train data
     * @param tstate              TrainState that will be used - updated
     * @return A node splitter generator
     */
    std::shared_ptr<tsc::i_GenNode> make_node_splitter(
            std::vector<F> const &exponents,
            std::vector<std::string> const &transforms,
            std::set<std::string> const &distances,
            size_t nbc,
            size_t series_max_length,
            std::map<std::string, tempo::DTS> const &train_data,
            tsc::TreeState &tstate
    );

}; // End of namespace pf::splitters
