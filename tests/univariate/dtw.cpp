#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <tempo/univariate/elastic_distances/dtw/dtw.hpp>

#include "tests_tools.hpp"
#include "references/dtw/dtw.hpp"

using namespace tempo::univariate::elastic_distances;
constexpr double POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<double>;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("DTW Fixed length", "[dtw]") {

    // Create a random dataset
    constexpr int nbitems = ttools::def_nbitems;
    constexpr int fixed = ttools::def_fixed_size;
    const auto fset = ttools::get_set_fixed_length(ttools::prng, nbitems, fixed);

    SECTION("DTW(s,s) == 0") {
        for (const auto &series: fset) {
            const double dtw_ref_v = reference::dtw_matrix(series, series);
            REQUIRE(dtw_ref_v == 0);

            const double dtw_v = dtw(series, series);
            REQUIRE(dtw_v == 0);

            const double dtw_eap_v = dtw(series, series, POSITIVE_INFINITY);
            REQUIRE(dtw_eap_v == 0);
        }
    }

    SECTION("DTW(s1, s2)") {
        for (int i = 0; i < nbitems; i += 2) {
            const auto &series1 = fset[i];
            const auto &series2 = fset[i + 1];

            const double dtw_ref_v = reference::dtw_matrix(series1, series2);
            INFO("Exact same operation order. Expect exact floating point equality.")

            const double dtw_v = dtw(series1, series2);
            REQUIRE(dtw_ref_v == dtw_v);

            const double dtw_eap_v = dtw(series1, series2, POSITIVE_INFINITY);
            REQUIRE(dtw_ref_v == dtw_eap_v);
        }
    }

    SECTION("NN1 DTW"){
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
                double v_ref = reference::dtw_matrix(fset[i], fset[j]);
                if (v_ref < bsf_ref) {
                    idx_ref = j;
                    bsf_ref = v_ref;
                }

                // --- --- --- --- --- --- --- --- --- --- --- ---
                double v = dtw(fset[i], fset[j]);
                if (v < bsf) {
                    idx = j;
                    bsf = v;
                }

                REQUIRE(idx_ref == idx);

                // --- --- --- --- --- --- --- --- --- --- --- ---
                double v_eap = dtw(fset[i], fset[j], bsf_eap);
                if (v_eap < bsf_eap) {
                    idx_eap = j;
                    bsf_eap = v_eap;
                }

                REQUIRE(idx_ref == idx_eap);
            }
        }// End query loop
    }// End section
}


TEST_CASE("DTW variable length", "[dtw]") {

    // Create a random dataset
    constexpr int nbitems = ttools::def_nbitems;
    const auto fset = ttools::get_set_variable_length(ttools::prng, nbitems);

    SECTION("DTW(s,s) == 0") {
        for (const auto &series: fset) {
            const double dtw_ref_v = reference::dtw_matrix(series, series);
            REQUIRE(dtw_ref_v == 0);

            const double dtw_v = dtw(series, series);
            REQUIRE(dtw_v == 0);

            const double dtw_eap_v = dtw(series, series, POSITIVE_INFINITY);
            REQUIRE(dtw_eap_v == 0);
        }
    }

    SECTION("DTW(s1, s2)") {
        for (int i = 0; i < nbitems; i += 2) {
            const auto& series1 = fset[i];
            const auto& series2 = fset[i + 1];

            const double dtw_ref_v = reference::dtw_matrix(series1, series2);
            INFO("Exact same operation order. Expect exact floating point equality.")

            const double dtw_v = dtw(series1, series2);
            REQUIRE(dtw_ref_v == dtw_v);

            const double dtw_eap_v = dtw(series1, series2, POSITIVE_INFINITY);
            REQUIRE(dtw_ref_v == dtw_eap_v);
        }
    }

    SECTION("NN1 DTW"){
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
                double v_ref = reference::dtw_matrix(fset[i], fset[j]);
                if (v_ref < bsf_ref) {
                    idx_ref = j;
                    bsf_ref = v_ref;
                }

                // --- --- --- --- --- --- --- --- --- --- --- ---
                double v = dtw(fset[i], fset[j]);
                if (v < bsf) {
                    idx = j;
                    bsf = v;
                }

                REQUIRE(idx_ref == idx);

                // --- --- --- --- --- --- --- --- --- --- --- ---
                double v_eap = dtw(fset[i], fset[j], bsf_eap);
                if (v_eap < bsf_eap) {
                    idx_eap = j;
                    bsf_eap = v_eap;
                }

                REQUIRE(idx_ref == idx_eap);
            }
        }// End query loop
    }// End Section
}
