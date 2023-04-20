#include <exception>

#include "pfsplitters.hpp"

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

namespace pf::splitters {

    using F = double;
    namespace tsc = tempo::classifier::TSChief;
    namespace tsc_nn1 = tempo::classifier::TSChief::snode::nn1splitter;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Parameterization
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    // --- --- --- Cost function exponent

    tsc_nn1::ExponentGetter make_get_vcfe(std::vector<F> exponent_set) {
        return [es = std::move(exponent_set)](tsc::TreeState &s) { return tempo::utils::pick_one(es, s.prng); };
    }

    tsc_nn1::ExponentGetter make_get_cfe1() { return [](tsc::TreeState & /*s*/) { return 1.0; }; }

    tsc_nn1::ExponentGetter make_get_cfe2() { return [](tsc::TreeState & /*s*/) { return 2.0; }; }

    // --- --- --- Transform getters

    tsc_nn1::TransformGetter make_get_transform(std::vector<std::string> tr_set) {
        return [ts = std::move(tr_set)](tsc::TreeState &s) { return tempo::utils::pick_one(ts, s.prng); };
    }

    tsc_nn1::TransformGetter make_get_default() {
        return [](tsc::TreeState & /*s*/) { return "default"; };
    }

    tsc_nn1::TransformGetter make_get_derivative(size_t d) {
        return [d](tsc::TreeState & /*s*/) { return "derivative" + std::to_string(d); };
    }

    // --- --- --- Window getter

    tsc_nn1::WindowGetter make_get_window(size_t maxlength) {
        return [=](tsc::TreeState &s, tsc::TreeData const & /* d */) {
            const size_t win_top = std::floor(((double) maxlength + 1) / 4.0);
            return std::uniform_int_distribution<size_t>(0, win_top)(s.prng);
        };
    }

    tsc_nn1::WindowGetter make_get_window(size_t maxlength, const double ratio) {
        const size_t win_top = std::floor((double) maxlength * ratio);
        return [=](tsc::TreeState &s, tsc::TreeData const & /* d */) {
            return std::uniform_int_distribution<size_t>(0, win_top)(s.prng);
        };
    }

    tsc_nn1::WindowGetter make_proba_window(size_t maxlength) {
        // Check arg
        if (maxlength < 2) { throw std::invalid_argument("maxlength < 2 (" + std::to_string(maxlength) + ")"); }

        // Generate weights - Note: proba i==0 never added
        // std::discrete_distribution produces random integer i in [0, n[ where i probability depends on a weight.
        // Example: for maxlength=4, we make the weights [3, 2, 1], resulting in distribution of integers [0, 1, 2],
        // where, '0' is three times more likely than '2' - actual chances are [3/6, 2/6, 1/6]
        // We can directly use the produced integer as window
        std::vector<double> weights;
        weights.reserve(maxlength - 1);
        for (size_t i = maxlength - 1; i == 0; --i) {
            weights.push_back((double) i);
        }

        // Note: for some reason, the '()' operator is not const, even if the doc states that
        // 'the associated parameter set is not modified' - we need the 'mutable' qualifier
        std::discrete_distribution<size_t> d(weights.begin(), weights.end());
        return [dd = std::move(d)](tsc::TreeState &s, tsc::TreeData const & /* d */) mutable {
            size_t r = dd(s.prng);
            return r;
        };

    }

    // --- --- --- ERP Gap Value *AND* LCSS epsilon.

    tsc_nn1::StatGetter make_get_frac_stddev() {
        return [=](tsc::TreeState &s, tsc::TreeData const &data, tempo::ByClassMap const &bcm,
                   std::string const &tr_name) {
            const tempo::DTS &train_dataset = tempo::classifier::TSChief::at_train(data).at(tr_name);
            auto stddev_ = stddev(train_dataset, bcm.to_IndexSet());
            return std::uniform_real_distribution<F>(stddev_ / 5.0, stddev_)(s.prng);
        };
    }

    tsc_nn1::StatGetter make_get_frac_stddev_0_stddev() {
        return [=](tsc::TreeState &s, tsc::TreeData const &data, tempo::ByClassMap const &bcm,
                   std::string const &tr_name) {
            const tempo::DTS &train_dataset = tempo::classifier::TSChief::at_train(data).at(tr_name);
            auto stddev_ = stddev(train_dataset, bcm.to_IndexSet());
            return std::uniform_real_distribution<F>(stddev_ / 5.0, stddev_)(s.prng);
        };
    }

    // --- --- --- MSM Cost

    tsc_nn1::T_GetterState<F> make_get_msm_cost() {
        return [](tsc::TreeState &state) {
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
    }

    // --- --- --- TWE nu & lambda parameters

    tsc_nn1::T_GetterState<F> make_get_twe_nu() {
        return [](tsc::TreeState &state) {
            constexpr size_t N = 10;
            constexpr F nus[N]{0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1};
            return tempo::utils::pick_one(nus, N, state.prng);
        };
    }

    tsc_nn1::T_GetterState<F> make_get_twe_lambda() {
        return [](tsc::TreeState &state) {
            constexpr size_t N = 10;
            constexpr F lambdas[N]{0, 0.011111111, 0.022222222, 0.033333333, 0.044444444,
                                   0.055555556, 0.066666667, 0.077777778, 0.088888889, 0.1};
            return tempo::utils::pick_one(lambdas, N, state.prng);
        };
    }


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Splitters
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    // --- --- --- Leaf Generator

    std::shared_ptr<tsc::i_GenLeaf> make_pure_leaf(tempo::DatasetHeader const &train_header) {
        return std::make_shared<tsc::sleaf::GenLeaf_Pure>(train_header);
    }

    std::shared_ptr<tsc::i_GenLeaf> make_pure_leaf_smoothp(tempo::DatasetHeader const &train_header) {
        return std::make_shared<tsc::sleaf::GenLeaf_PureSmoothP>(train_header);
    }

    std::shared_ptr<tsc::i_GenNode> make_node_splitter(
            std::vector<F> const &exponents,
            std::vector<std::string> const &transforms,
            std::set<std::string> const &distances,
            size_t nbc,
            size_t series_max_length,
            std::map<std::string, tempo::DTS> const &train_data,
            tsc::TreeState &tstate
    ) {

        // --- --- --- State
        // 1NN distance splitters - cache the indexset
        using GS1NNState = tsc_nn1::GenSplitterNN1_State;
        std::shared_ptr<tsc::i_GetState<GS1NNState>> get_GenSplitterNN1_State =
                tstate.register_state<GS1NNState>(std::make_unique<GS1NNState>());

        // --- --- --- Getters

        // --- --- --- Build distance generators

        // List of distance generators (this is specific to our collection of NN1 Splitter generators)
        std::vector<std::shared_ptr<tsc_nn1::i_GenDist>> gendist;

        if (distances.empty()) { throw std::invalid_argument("Empty set of distances"); }

        // --- --- --- PF2018
        if (*distances.begin() == "pf2018") {

            auto getter_cfe_2 = make_get_cfe2();
            auto getter_tr_def = make_get_default();
            auto getter_tr_dr1 = make_get_derivative(1);
            auto getter_window = make_get_window(series_max_length);
            auto frac_stddev = make_get_frac_stddev();
            auto getter_msm_cost = make_get_msm_cost();
            auto getter_twe_nu = make_get_twe_nu();
            auto getter_twe_lambda = make_get_twe_lambda();

            // default ED
            gendist.push_back(make_shared<tsc_nn1::DAGen>(getter_tr_def, getter_cfe_2));

            // default DTW
            gendist.push_back(make_shared<tsc_nn1::DTWGen>(getter_tr_def, getter_cfe_2, getter_window));
            // derivative 1 DTW
            gendist.push_back(make_shared<tsc_nn1::DTWGen>(getter_tr_dr1, getter_cfe_2, getter_window));

            // default DTWFull
            gendist.push_back(make_shared<tsc_nn1::DTWFullGen>(getter_tr_def, getter_cfe_2));
            // derivative 1 DTWFull
            gendist.push_back(make_shared<tsc_nn1::DTWFullGen>(getter_tr_dr1, getter_cfe_2));

            // default WDTW
            gendist.push_back(make_shared<tsc_nn1::WDTWGen>(getter_tr_def, getter_cfe_2, series_max_length));
            // derivative 1 WDTW
            gendist.push_back(make_shared<tsc_nn1::WDTWGen>(getter_tr_dr1, getter_cfe_2, series_max_length));

            // ERP
            gendist.push_back(make_shared<tsc_nn1::ERPGen>(getter_tr_def, getter_cfe_2, frac_stddev, getter_window));

            // LCSS
            gendist.push_back(make_shared<tsc_nn1::LCSSGen>(getter_tr_def, frac_stddev, getter_window));

            // MSM
            gendist.push_back(make_shared<tsc_nn1::MSMGen>(getter_tr_def, getter_msm_cost));

            // TWE
            gendist.push_back(make_shared<tsc_nn1::TWEGen>(getter_tr_def, getter_twe_nu, getter_twe_lambda));
        } else if (distances.contains("pf2")) {
            // --- --- --- PF2.0
            auto getter_cfe_set = make_get_vcfe(exponents);
            auto getter_tr_set = make_get_transform(transforms);
            auto getter_window = make_get_window(series_max_length);
            auto frac_stddev = make_get_frac_stddev();

            // ADTW
            // Sample train data
            constexpr size_t SAMPLE_SIZE = 4000;
            auto samples = tsc_nn1::ADTWGen::do_sampling(exponents, transforms, train_data, SAMPLE_SIZE, tstate.prng);
            // Create distance
            gendist.push_back(make_shared<tsc_nn1::ADTWGen>(getter_tr_set, getter_cfe_set, samples));

            // DTW
            if (distances.contains("dtwproba")) {
                // Proba way
                auto getter_window_proba = make_proba_window(series_max_length);
                gendist.push_back(make_shared<tsc_nn1::DTWGen>(getter_tr_set, getter_cfe_set, getter_window_proba));
            } else {
                // Classic way
                gendist.push_back(make_shared<tsc_nn1::DTWGen>(getter_tr_set, getter_cfe_set, getter_window));
            }

            // LCSS
            gendist.push_back(make_shared<tsc_nn1::LCSSGen>(getter_tr_set, frac_stddev, getter_window));

        } else {
            // --- --- --- Any other combination with cost function 2
            auto getter_cfe_set = make_get_vcfe(exponents);
            auto getter_cfe_2 = make_get_cfe2();
            auto getter_tr_set = make_get_transform(transforms);
            auto getter_window = make_get_window(series_max_length);
            auto frac_stddev = make_get_frac_stddev();
            auto getter_msm_cost = make_get_msm_cost();
            auto getter_twe_nu = make_get_twe_nu();
            auto getter_twe_lambda = make_get_twe_lambda();

            for (std::string const &sname: distances) {

                if (sname.starts_with("DA")) {
                    // --- --- --- Direct Alignment
                    gendist.push_back(make_shared<tsc_nn1::DAGen>(getter_tr_set, getter_cfe_set));
                } else if (sname.starts_with("ADTW")) {
                    // --- --- --- ADTW
                    // Sample train data
                    constexpr size_t SAMPLE_SIZE = 4000;
                    auto samples = tsc_nn1::ADTWGen::do_sampling(exponents, transforms, train_data, SAMPLE_SIZE,
                                                                 tstate.prng);
                    // Create distance
                    gendist.push_back(make_shared<tsc_nn1::ADTWGen>(getter_tr_set, getter_cfe_set, samples));
                } else if (sname.starts_with("DTW") && !sname.starts_with("DTWFull")) {
                    // --- --- --- DTW
                    gendist.push_back(make_shared<tsc_nn1::DTWGen>(getter_tr_set, getter_cfe_set, getter_window));
                } else if (sname.starts_with("WDTW")) {
                    // --- --- --- WDTW
                    gendist.push_back(make_shared<tsc_nn1::WDTWGen>(getter_tr_set, getter_cfe_set, series_max_length));
                } else if (sname.starts_with("DTWFull")) {
                    // --- --- --- DTWFull
                    gendist.push_back(make_shared<tsc_nn1::DTWFullGen>(getter_tr_set, getter_cfe_set));
                } else if (sname.starts_with("ERP")) {
                    // --- --- --- ERP
                    gendist.push_back(
                            make_shared<tsc_nn1::ERPGen>(getter_tr_set, getter_cfe_2, frac_stddev, getter_window));
                } else if (sname.starts_with("LCSS")) {
                    // --- --- --- LCSS
                    gendist.push_back(make_shared<tsc_nn1::LCSSGen>(getter_tr_set, frac_stddev, getter_window));
                } else if (sname.starts_with("MSM")) {
                    // --- --- --- MSM
                    gendist.push_back(make_shared<tsc_nn1::MSMGen>(getter_tr_set, getter_msm_cost));
                } else if (sname.starts_with("TWE")) {
                    // --- --- --- TWE
                    gendist.push_back(make_shared<tsc_nn1::TWEGen>(getter_tr_set, getter_twe_nu, getter_twe_lambda));
                }
            }
        }

        // Build vector for the node generator
        std::vector<std::shared_ptr<tsc::i_GenNode>> generators;

        // Wrap each distance generator in GenSplitter1NN (which is a i_GenNode) and push in generators
        for (auto const &gd: gendist) {
            generators.push_back(
                    make_shared<tsc_nn1::GenSplitterNN1>(gd,
                                                         get_GenSplitterNN1_State) //, get_train_data, get_test_data)
            );
        }

        // --- Put a node chooser over all generators
        return make_shared<tsc::snode::meta::SplitterChooserGen>(std::move(generators), nbc);
    }

}; // End of namespace pf::splitters