#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <tempo/univariate/distances/dtw/wdtw.hpp>

#include "tests_tools.hpp"
#include "references/dtw/wdtw.hpp"

using namespace tempo::univariate::distances;
constexpr double POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<double>;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

TEST_CASE("Test weights generation", "[wdtw]"){
    constexpr size_t maxlength = 50;
    constexpr double mc = ((double) maxlength)/2;
    for(double g: ttools::weight_factors) {
        auto weights = generate_weights(g, maxlength);
        for(size_t i=0; i < maxlength; ++i){
            auto lib = weights[i];
            auto ref = reference::mlwf(i, mc, g);
            REQUIRE(lib==ref);
        }
    }
}


TEST_CASE("WDTW Fixed length", "[wdtw]") {

    // Create a random dataset
    constexpr int nbitems = ttools::def_nbitems;
    constexpr int fixed = ttools::def_fixed_size;
    const auto fset = ttools::get_set_fixed_length(ttools::prng, nbitems, fixed);

    SECTION("WDTW(s,s) == 0") {
        for(double g: ttools::weight_factors) {
            auto weights = generate_weights(g, fixed);

            for (const auto &series: fset) {

                const double wdtw_ref_v = reference::wdtw_matrix(series, series, weights);
                REQUIRE(wdtw_ref_v == 0);

                const double wdtw_v = wdtw(series, series, weights);
                REQUIRE(wdtw_v == 0);

                const double wdtw_eap_v = wdtw(series, series, weights, POSITIVE_INFINITY);
                REQUIRE(wdtw_eap_v == 0);
            }
        }
    }

    SECTION("WDTW(s1, s2)") {
        for(double g: ttools::weight_factors) {
            auto weights = generate_weights(g, fixed);

            for (int i = 0; i < nbitems; i += 2) {
                const auto &series1 = fset[i];
                const auto &series2 = fset[i + 1];

                const double wdtw_ref_v = reference::wdtw_matrix(series1, series2, weights);
                INFO("Exact same operation order. Expect exact floating point equality.")

                const double wdtw_v = wdtw(series1, series2, weights);
                REQUIRE(wdtw_ref_v == wdtw_v);

                const double wdtw_eap_v = wdtw(series1, series2, weights, POSITIVE_INFINITY);
                REQUIRE(wdtw_ref_v == wdtw_eap_v);
            }
        }
    }

    SECTION("NN1 WDTW"){
        // Weights loop
        for(double g: ttools::weight_factors) {
            auto weights = generate_weights(g, fixed);

            // Query loop
            for (int i = 0; i < nbitems; i += 3) {

                // Ref Variables
                int idx_ref = 0;
                double bsf_ref = POSITIVE_INFINITY;

                // Base Variables
                int idx = 0;
                double bsf = POSITIVE_INFINITY;

                // EAP Variables
                int idx_eap = 0;
                double bsf_eap = POSITIVE_INFINITY;

                // NN1 loop
                for (int j = 0; j < nbitems; j += 5) {
                    // Skip self.
                    if (i == j) { continue; }

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    double v_ref = reference::wdtw_matrix(fset[i], fset[j], weights);
                    if (v_ref < bsf_ref) {
                        idx_ref = j;
                        bsf_ref = v_ref;
                    }

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    double v = wdtw(fset[i], fset[j], weights);
                    if (v < bsf) {
                        idx = j;
                        bsf = v;
                    }

                    REQUIRE(idx_ref == idx);

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    double v_eap = wdtw(fset[i], fset[j], weights, bsf_eap);
                    if (v_eap < bsf_eap) {
                        idx_eap = j;
                        bsf_eap = v_eap;
                    }

                    REQUIRE(idx_ref == idx_eap);
                }
            }// End query loop
        } // End weights loop
    }// End section
}


TEST_CASE("WDTW variable length", "[wdtw]") {

    // Create a random dataset
    constexpr int nbitems = ttools::def_nbitems;
    constexpr size_t maxsize = ttools::def_max_size;
    const auto fset = ttools::get_set_variable_length(ttools::prng, nbitems, 0, maxsize);

    SECTION("WDTW(s,s) == 0") {
        for(double g: ttools::weight_factors) {
            auto weights = generate_weights(g, maxsize);

            for (const auto &series: fset) {

                const double wdtw_ref_v = reference::wdtw_matrix(series, series, weights);
                REQUIRE(wdtw_ref_v == 0);

                const double wdtw_v = wdtw(series, series, weights);
                REQUIRE(wdtw_v == 0);

                const double wdtw_eap_v = wdtw(series, series, weights, POSITIVE_INFINITY);
                REQUIRE(wdtw_eap_v == 0);
            }
        }
    }

    SECTION("WDTW(s1, s2)") {

        for(double g: ttools::weight_factors) {
            auto weights = generate_weights(g, maxsize);

            for (int i = 0; i < nbitems; i += 2) {
                const auto& series1 = fset[i];
                const auto& series2 = fset[i + 1];

                const double wdtw_ref_v = reference::wdtw_matrix(series1, series2, weights);
                INFO("Exact same operation order. Expect exact floating point equality.")

                const double wdtw_v = wdtw(series1, series2, weights);
                REQUIRE(wdtw_ref_v == wdtw_v);

                const double wdtw_eap_v = wdtw(series1, series2, weights, POSITIVE_INFINITY);
                REQUIRE(wdtw_ref_v == wdtw_eap_v);
            }
        }
    }

    SECTION("NN1 WDTW"){

        // Weights loop
        for(double g: ttools::weight_factors) {
            auto weights = generate_weights(g, maxsize);

            // Query loop
            for(int i=0; i<nbitems; i+=3) {

                // Ref Variables
                int idx_ref=0;
                double bsf_ref = POSITIVE_INFINITY;

                // Base Variables
                int idx=0;
                double bsf = POSITIVE_INFINITY;

                // EAP Variables
                int idx_eap = 0;
                double bsf_eap = POSITIVE_INFINITY;

                // NN1 loop
                for (int j = 0; j < nbitems; j+=5) {
                    // Skip self.
                    if(i==j){continue;}

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    double v_ref = reference::wdtw_matrix(fset[i], fset[j], weights);
                    if (v_ref < bsf_ref) {
                        idx_ref = j;
                        bsf_ref = v_ref;
                    }

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    double v = wdtw(fset[i], fset[j], weights);
                    if (v < bsf) {
                        idx = j;
                        bsf = v;
                    }
                    REQUIRE(idx_ref == idx);

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    double v_eap = wdtw(fset[i], fset[j], weights, bsf_eap);
                    if (v_eap < bsf_eap) {
                        idx_eap = j;
                        bsf_eap = v_eap;
                    }

                    REQUIRE(idx_ref == idx_eap);
                }
            }// End query loop
        }// End Weights loop
    }// End Section
}
