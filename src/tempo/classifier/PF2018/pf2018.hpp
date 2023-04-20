#pragma once

#include <tempo/dataset/dts.hpp>
#include <tempo/classifier/TSChief/splitter_interface.hpp>
#include <tempo/classifier/TSChief/snode/nn1splitter/nn1dist_interface.hpp>
#include <tempo/classifier/TSChief/snode/nn1splitter/nn1splitter.hpp>

#include "tempo/classifier/TSChief/sleaf/pure_leaf.hpp"

#include "tempo/classifier/TSChief/tree.hpp"
#include "tempo/classifier/TSChief/forest.hpp"
#include "tempo/classifier/TSChief/sleaf/pure_leaf.hpp"
#include "tempo/classifier/TSChief/sleaf/pure_leaf_smoothp.hpp"
#include "tempo/classifier/TSChief/snode/meta/chooser.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1splitter.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_directa.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_adtw.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_dtw.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_dtwfull.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_wdtw.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_erp.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_lcss.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_msm.hpp"
#include "tempo/classifier/TSChief/snode/nn1splitter/nn1_twe.hpp"

namespace tempo::classifier {

    namespace tsc = tempo::classifier::TSChief;
    namespace tsc_nn1 = tempo::classifier::TSChief::snode::nn1splitter;

    class PF2018 {

        // --- From constructor
        DatasetHeader const &train_header;
        const size_t nb_candidates;
        const size_t nb_trees;


    public:

        PF2018(DatasetHeader const &train_header, size_t nb_candidates, size_t nb_trees, tsc::TreeState &tstate) :
                train_header(train_header), nb_candidates(nb_candidates), nb_trees(nb_trees) {
        }

        tsc::ForestTrainer forest_trainer(train_header, tree_trainer, opt.nb_trees);

        // --- --- --- TRAIN

        void train(tsc::TreeState &tstate) {

            std::shared_ptr<tsc::TreeTrainer> tree_trainer;

            size_t maxlength = train_header.length_max();

            // --- --- --- Build the node generator
            std::shared_ptr<tsc::i_GenNode> node_gen;
            {
                // Configure distance with parameterization functions
                std::vector<std::shared_ptr<tsc_nn1::i_GenDist>> gendist;
                {

                    // --- --- --- Parameters

                    // Cost Function Exponent: always 2
                    tsc_nn1::ExponentGetter cfe2 = [](tsc::TreeState &) -> double { return 2.0; };

                    // Transforms - predefined 'default' and 'derivative1'
                    tsc_nn1::TransformGetter tr_def = [](tsc::TreeState &) -> std::string { "default"; };
                    tsc_nn1::TransformGetter tr_d1 = [](tsc::TreeState &) -> std::string { "derivative1"; };

                    // Warping window: sample in [0, max_size/4]
                    tsc_nn1::WindowGetter get_window = [maxlength](tsc::TreeState &s,
                                                                   tsc::TreeData const & /* d */) -> size_t {
                        const size_t win_top = std::floor((double) maxlength / 4.0);
                        return std::uniform_int_distribution<size_t>(0, win_top)(s.prng);
                    };

                    // Sample in [0.2*stddev, stddev[
                    tsc_nn1::StatGetter get_frac_stddev = [](
                            tsc::TreeState &s,
                            tsc::TreeData const &data,
                            tempo::ByClassMap const &bcm,
                            std::string const &tr_name
                    ) {
                        const tempo::DTS &train_dataset = tsc::at_train(data).at(tr_name);
                        auto stddev_ = stddev(train_dataset, bcm.to_IndexSet());
                        return std::uniform_real_distribution<F>(stddev_ / 5.0, stddev_)(s.prng);
                    };

                    // MSM cost
                    tsc_nn1::T_GetterState<F> get_msm_cost = [](tsc::TreeState &state) {
                        constexpr size_t MSM_N = 100;
                        constexpr F msm_cost[MSM_N]{
                                0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325, 0.03625, 0.04, 0.04375,
                                0.0475, 0.05125, 0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375, 0.0775, 0.08125,
                                0.085, 0.08875, 0.0925, 0.09625, 0.1, 0.136, 0.172, 0.208, 0.244, 0.28, 0.316, 0.352,
                                0.388, 0.424, 0.46, 0.496, 0.532, 0.568, 0.604, 0.64, 0.676, 0.712, 0.748, 0.784,
                                0.82, 0.856, 0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8, 3.16, 3.52, 3.88,
                                4.24, 4.6, 4.96, 5.32, 5.68, 6.04, 6.4, 6.76, 7.12, 7.48, 7.84, 8.2, 8.56, 8.92, 9.28,
                                9.64, 10, 13.6, 17.2, 20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46, 49.6, 53.2, 56.8,
                                60.4, 64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4, 100
                        };
                        return tempo::utils::pick_one(msm_cost, MSM_N, state.prng);
                    };

                    // TWE NU
                    tsc_nn1::T_GetterState<F> get_twe_nu = [](tsc::TreeState &state) {
                        constexpr size_t N = 10;
                        constexpr F nus[N]{0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1};
                        return tempo::utils::pick_one(nus, N, state.prng);
                    };

                    // TWE Lambda
                    tsc_nn1::T_GetterState<F> get_twe_lambda = [](tsc::TreeState &state) {
                        constexpr size_t N = 10;
                        constexpr F lambdas[N]{0, 0.011111111, 0.022222222, 0.033333333, 0.044444444,
                                               0.055555556, 0.066666667, 0.077777778, 0.088888889, 0.1};
                        return tempo::utils::pick_one(lambdas, N, state.prng);
                    };


                    // default ED
                    gendist.push_back(make_shared<tsc_nn1::DAGen>(tr_def, cfe2));

                    // default DTW
                    gendist.push_back(make_shared<tsc_nn1::DTWGen>(tr_def, cfe2, get_window));
                    // derivative 1 DTW
                    gendist.push_back(make_shared<tsc_nn1::DTWGen>(tr_d1, cfe2, get_window));

                    // default DTWFull
                    gendist.push_back(make_shared<tsc_nn1::DTWFullGen>(tr_def, cfe2));
                    // derivative 1 DTWFull
                    gendist.push_back(make_shared<tsc_nn1::DTWFullGen>(tr_d1, cfe2));

                    // default WDTW
                    gendist.push_back(make_shared<tsc_nn1::WDTWGen>(tr_def, cfe2, maxlength));
                    // derivative 1 WDTW
                    gendist.push_back(make_shared<tsc_nn1::WDTWGen>(tr_d1, cfe2, maxlength));

                    // ERP
                    gendist.push_back(make_shared<tsc_nn1::ERPGen>(tr_def, cfe2, get_frac_stddev, get_window));

                    // LCSS
                    gendist.push_back(make_shared<tsc_nn1::LCSSGen>(tr_def, get_frac_stddev, get_window));

                    // MSM
                    gendist.push_back(make_shared<tsc_nn1::MSMGen>(tr_def, get_msm_cost));

                    // TWE
                    gendist.push_back(make_shared<tsc_nn1::TWEGen>(tr_def, get_twe_nu, get_twe_lambda));
                }

                // Wrap distances into Node Generators

                // State access: record a cache for the BCM to IndexSet computation
                std::shared_ptr<tsc::i_GetState<tsc_nn1::GenSplitterNN1_State>> get_GenSplitterNN1_State =
                        tstate.build_state<tsc_nn1::GenSplitterNN1_State>();

                // Build vector for the node generator
                std::vector<std::shared_ptr<tsc::i_GenNode>> generators;

                // Wrap each distance generator in GenSplitter1NN (which is a i_GenNode) and push in generators
                for (auto const &gd: gendist) {
                    generators.push_back(make_shared<tsc_nn1::GenSplitterNN1>(gd, get_GenSplitterNN1_State));
                }

                // Random chooser over the node generators
                node_gen = make_shared<tsc::snode::meta::SplitterChooserGen>(std::move(generators), nb_candidates);
            }

            // --- --- --- Build the leaf generator
            std::shared_ptr<tsc::i_GenLeaf> leaf_gen;
            {
                leaf_gen = std::make_shared<tsc::sleaf::GenLeaf_Pure>(train_header);
            }

            // --- --- --- Make the tree trainer
            tree_trainer = std::make_shared<tsc::TreeTrainer>(leaf_gen, node_gen);


        }

        auto train_start_time = utils::now();
        auto forest = forest_trainer.train(tstate, tdata, train_bcm, opt.nb_threads, &std::cout);
        auto train_elapsed = utils::now() - train_start_time;

        train

    }; // End of struct PF2018

}; // End of namespace tempo::classifier
