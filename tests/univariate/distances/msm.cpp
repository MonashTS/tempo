#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <tempo/univariate/distances/msm/msm.hpp>

#include "../tests_tools.hpp"
#include "references/msm/msm.hpp"

using namespace tempo::univariate;
constexpr double POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<double>;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

TEST_CASE("MSM Fixed length", "[msm]") {

    // Create a random dataset
    constexpr int nbitems = ttools::def_nbitems;
    constexpr int fsize = ttools::def_fixed_size;
    const auto fset = ttools::get_set_fixed_length(ttools::prng, nbitems, fsize);

    SECTION("MSM(s,s) == 0") {
        for (auto c: ttools::msm_costs) {
            for (const auto &series: fset) {
                double vref = reference::msm_matrix(series, series, c);
                REQUIRE(vref == 0);

                double v = msm(series, series, c);
                REQUIRE(v == 0);

                double v_eap = msm(series, series, c, POSITIVE_INFINITY);
                REQUIRE(v_eap == 0);
            }
        }
    }

    SECTION("MSM(s1, s2)") {
        for (auto c: ttools::msm_costs) {
            for (int i = 0; i < nbitems; i += 2) {
                const auto &series1 = fset[i];
                const auto &series2 = fset[i + 1];

                double vref = reference::msm_matrix(series1, series2, c);
                INFO("Exact same operation order. Expect exact floating point equality.")

                double v = msm(series1, series2, c);
                REQUIRE(vref == v);

                double v_eap = msm(series1, series2, c, POSITIVE_INFINITY);
                REQUIRE(vref == v_eap);
            }
        }
    }

    SECTION("NN1 MSM") {

        for (auto c: ttools::msm_costs) {
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
                    double v_ref = reference::msm_matrix(fset[i], fset[j], c);
                    if (v_ref < bsf_ref) {
                        idx_ref = j;
                        bsf_ref = v_ref;
                    }

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    double v = msm(fset[i], fset[j], c);
                    if (v < bsf) {
                        idx = j;
                        bsf = v;
                    }

                    REQUIRE(idx_ref == idx);

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    double v_eap = msm(fset[i], fset[j], c, bsf_eap);
                    if (v_eap < bsf_eap) {
                        idx_eap = j;
                        bsf_eap = v_eap;
                    }

                    REQUIRE(idx_ref == idx_eap);
                }
            }// End query loop
        }
    }// End section
}


TEST_CASE("MSM variable length", "[msm]") {

    // Create a random dataset
    constexpr int nbitems = ttools::def_nbitems;
    const auto fset = ttools::get_set_variable_length(ttools::prng, nbitems, 0, 50);

    SECTION("MSM(s,s) == 0") {
        for (auto c: ttools::msm_costs) {
            for (const auto &series: fset) {
                double vref = reference::msm_matrix(series, series, c);
                REQUIRE(vref == 0);

                double v = msm(series, series, c);
                REQUIRE(v == 0);

                double v_eap = msm(series, series, c, POSITIVE_INFINITY);
                REQUIRE(v_eap == 0);
            }
        }
    }

    SECTION("MSM(s1, s2)") {
        for (auto c: ttools::msm_costs) {
            for (int i = 0; i < nbitems; i += 2) {
                const auto &series1 = fset[i];
                const auto &series2 = fset[i + 1];
                double vref = reference::msm_matrix(series1, series2, c);

                double v = msm(series1, series2, c);
                REQUIRE(vref == v);

                double v_eap = msm(series1, series2, c, POSITIVE_INFINITY);
                REQUIRE(vref == v_eap);
            }
        }
    }

    SECTION("NN1 MSM") {

        for (auto c: ttools::msm_costs) {
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
                    double v_ref = reference::msm_matrix(fset[i], fset[j], c);
                    if (v_ref < bsf_ref) {
                        idx_ref = j;
                        bsf_ref = v_ref;
                    }

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    double v = msm(fset[i], fset[j], c);
                    if (v < bsf) {
                        idx = j;
                        bsf = v;
                    }

                    REQUIRE(idx_ref == idx);

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    double v_eap = msm(fset[i], fset[j], c, bsf_eap);
                    if (v_eap < bsf_eap) {
                        idx_eap = j;
                        bsf_eap = v_eap;
                    }

                    REQUIRE(idx_ref == idx_eap);
                }
            }// End query loop
        }
    }// End section
}
