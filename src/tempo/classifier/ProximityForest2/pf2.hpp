#pragma once

#include <regex>
#include <tempo/dataset/dts.hpp>
#include <tempo/reader/dts.reader.hpp>
#include <tempo/transform/tseries.univariate.hpp>
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

#include "tempo/classifier/TSChief/pfsplitters.hpp"

namespace tempo::classifier {

    using namespace std;
    using namespace tempo::reader::dataset;
    using MDTS = std::map<std::string, tempo::DTS>;

    namespace tsc = tempo::classifier::TSChief;
    namespace tsc_nn1 = tempo::classifier::TSChief::snode::nn1splitter;

    namespace ttu = tempo::transform::univariate;

    class ProximityForest2 {

        // --- From constructor
        DTS const &train_dataset;
        DatasetHeader const &train_header;

        const size_t nb_candidates;
        const size_t nb_trees;

        const std::string str = "pf2";

        const std::string tr_default = "default";
        const std::string tr_d1 = "derivative1";

        const std::vector<std::string> transforms{tr_default, tr_d1};

        const std::vector<F> exponents{0.5, 1, 2};

        shared_ptr<MDTS> train_map = make_shared<MDTS>();
        shared_ptr<MDTS> test_map = make_shared<MDTS>();

        std::shared_ptr<tsc::Forest> forest;
        tsc::TreeData tdata;
        tsc::TreeState &tstate;

    public:
        ProximityForest2(
                DTS const &train_dataset,
                DatasetHeader const &train_header,
                size_t nb_candidates,
                size_t nb_trees,
                tsc::TreeState &tstate
        ) : train_dataset(train_dataset), train_header(train_header),
            nb_candidates(nb_candidates),
            nb_trees(nb_trees), tstate(tstate) {}


        utils::duration_t prepare_train_data_time;
        utils::duration_t prepare_test_data_time;
        utils::duration_t train_time;
        utils::duration_t test_time;

        // --- --- --- TRAIN

        void train(int nb_threads) {
            auto [train_bcm, train_bcm_remains] = train_dataset.get_BCM();

            // --- --- --- Prepare the data

            auto prepare_data_start_time = utils::now();
            {
                auto train_derive_t1 = train_dataset.transform().map_shptr<TSeries>(
                        [](TSeries const &t) { return ttu::derive(t); }, tr_d1);

                DTS train_derive_1("train", train_derive_t1);
                train_map->emplace(tr_default, train_dataset);
                train_map->emplace(tr_d1, train_derive_1);
            }
            prepare_train_data_time = utils::now() - prepare_data_start_time;

            tsc::register_train(tdata, train_map);


            // --- --- --- Build the leaf generator
            std::shared_ptr<tsc::i_GenLeaf> leaf_gen = pf::splitters::make_pure_leaf(train_header);

            regex r(":");
            std::set<std::string> distances(
                    sregex_token_iterator(str.begin(), str.end(), r, -1),
                    sregex_token_iterator()
            );
            if (distances.empty()) { throw std::invalid_argument("No distances registered (" + str + ")"); }

            // --- --- --- Build the node generator
            std::shared_ptr<tsc::i_GenNode> node_gen = pf::splitters::make_node_splitter(
                    exponents, transforms, distances, nb_candidates,
                    train_header.length_max(),
                    *train_map,
                    tstate
            );

            // --- --- --- Make the tree trainer
            std::shared_ptr<tsc::TreeTrainer> tree_trainer = std::make_shared<tsc::TreeTrainer>(leaf_gen, node_gen);

            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            // Use the forest
            // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            tsc::ForestTrainer forest_trainer(train_header, tree_trainer, nb_trees);

            std::optional<double> ratio = std::nullopt;

            auto train_start_time = utils::now();
            forest = forest_trainer.train(
                    tstate,
                    tdata,
                    train_bcm,
                    nb_threads,
                    ratio,
                    &std::cout
            );
            train_time = utils::now() - train_start_time;
        }

        classifier::ResultN predict(DTS const &test_dataset, int nb_threads) {
            auto prepare_data_start_time = utils::now();
            {
                auto test_derive_t1 = test_dataset.transform().map_shptr<TSeries>(
                        [](TSeries const &t) { return ttu::derive(t); }, tr_d1);
                DTS test_derive_1("test", test_derive_t1);
                test_map->emplace(tr_default, test_dataset);
                test_map->emplace(tr_d1, test_derive_1);
            }
            prepare_test_data_time = utils::now() - prepare_data_start_time;

            tsc::register_test(tdata, test_map);

            classifier::ResultN result;

            auto test_start_time = utils::now();
            const size_t test_size = test_dataset.size();
            tempo::utils::ProgressMonitor pm(test_size);

            for (size_t test_idx = 0; test_idx < test_size; ++test_idx) {
                // Get the prediction per tree
                std::vector<classifier::Result1> vecr = forest->predict(
                        tstate, tdata, test_idx, nb_threads
                );
                // Merge prediction as we want. Here, arithmetic average weighted by number of leafs
                // Result1 must be initialised with the number of classes!
                classifier::Result1 r1(train_header.nb_classes());
                for (const auto &r: vecr) {
                    r1.probabilities += r.probabilities * r.weight;
                    r1.weight += r.weight;
                }
                r1.probabilities /= r1.weight;
                //
                result.append(r1);
                //
                pm.print_progress(std::cout, test_idx);
            }
            std::cout << std::endl;
            test_time = utils::now() - test_start_time;

            return result;
        }
    }; // End of struct PF2


}; // End of namespace tempo::classifier
