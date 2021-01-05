#include <catch.hpp>
#include <tempo/univariate/distances/twe/twe.hpp>

#include "../tests_tools.hpp"
#include "references/twe/twe.hpp"

using namespace tempo::univariate;
constexpr double POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<double>;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

TEST_CASE("TWE Fixed length", "[twe]") {

    // Create a random dataset
    constexpr int nbitems = ttools::def_nbitems;
    constexpr int fsize = ttools::def_fixed_size;
    const auto fset = ttools::get_set_fixed_length(ttools::prng, nbitems, fsize);

    SECTION("TWE(s,s) == 0") {
        for (auto nu: ttools::twe_nus) {
            for (auto lambda: ttools::twe_lambdas) {
                for (const auto &series: fset) {
                    double vref = reference::twe_Marteau(series, series, nu, lambda);
                    REQUIRE(vref == 0);

                    double v = twe(series, series, nu, lambda);
                    REQUIRE(v == 0);

                    double v_eap = twe(series, series, nu, lambda, POSITIVE_INFINITY);
                    REQUIRE(v_eap == 0);
                }
            }
        }
    }

    SECTION("TWE(s1, s2)") {
        for (auto nu: ttools::twe_nus) {
            for (auto lambda: ttools::twe_lambdas) {
                for (int i = 0; i < nbitems; i += 2) {
                    const auto &series1 = fset[i];
                    const auto &series2 = fset[i + 1];

                    double vref = reference::twe_Marteau(series1, series2, nu, lambda);
                    INFO("Not exact same operation orders. Requires approximative equality.")

                    double v = twe(series1, series2, nu, lambda);
                    REQUIRE(vref == Approx(v));

                    double v_eap = twe(series1, series2, nu, lambda, POSITIVE_INFINITY);
                    REQUIRE(v == v_eap);
                }
            }
        }
    }

    SECTION("NN1 TWE") {

        for (auto nu: ttools::twe_nus) {
            for (auto lambda: ttools::twe_lambdas) {
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
                        double v_ref = reference::twe_Marteau(fset[i], fset[j], nu, lambda);
                        if (v_ref < bsf_ref) {
                            idx_ref = j;
                            bsf_ref = v_ref;
                        }

                        // --- --- --- --- --- --- --- --- --- --- --- ---
                        double v = twe(fset[i], fset[j], nu, lambda);
                        if (v < bsf) {
                            idx = j;
                            bsf = v;
                        }

                        REQUIRE(idx_ref == idx);

                        // --- --- --- --- --- --- --- --- --- --- --- ---
                        double v_eap = twe(fset[i], fset[j], nu, lambda, bsf_eap);
                        if (v_eap < bsf_eap) {
                            idx_eap = j;
                            bsf_eap = v_eap;
                        }

                        REQUIRE(idx_ref == idx_eap);
                    }
                }// End query loop
            }
        }
    }// End section
}


TEST_CASE("TWE variable length", "[twe]") {

    // Create a random dataset
    constexpr int nbitems = ttools::def_nbitems;
    const auto fset = ttools::get_set_variable_length(ttools::prng, nbitems);

    SECTION("TWE(s,s) == 0") {
        for (auto nu: ttools::twe_nus) {
            for (auto lambda: ttools::twe_lambdas) {
                for (const auto &series: fset) {
                    double vref = reference::twe_Marteau(series, series, nu, lambda);
                    REQUIRE(vref == 0);

                    double v = twe(series, series, nu, lambda);
                    REQUIRE(v == 0);

                    double v_eap = twe(series, series, nu, lambda, POSITIVE_INFINITY);
                    REQUIRE(v_eap == 0);
                }
            }
        }
    }

    SECTION("TWE(s1, s2)") {
        for (auto nu: ttools::twe_nus) {
            for (auto lambda: ttools::twe_lambdas) {
                for (int i = 0; i < nbitems; i += 2) {
                    const auto &series1 = fset[i];
                    const auto &series2 = fset[i + 1];

                    double vref = reference::twe_Marteau(series1, series2, nu, lambda);
                    INFO("Not exact same operation orders. Requires approximated equality.")

                    double v = twe(series1, series2, nu, lambda);
                    REQUIRE(vref == Approx(v));

                    double v_eap = twe(series1, series2, nu, lambda, POSITIVE_INFINITY);
                    REQUIRE(v == v_eap);
                }
            }
        }
    }

    SECTION("NN1 TWE") {

        for (auto nu: ttools::twe_nus) {
            for (auto lambda: ttools::twe_lambdas) {
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
                        double v_ref = reference::twe_Marteau(fset[i], fset[j], nu, lambda);
                        if (v_ref < bsf_ref) {
                            idx_ref = j;
                            bsf_ref = v_ref;
                        }

                        // --- --- --- --- --- --- --- --- --- --- --- ---
                        double v = twe(fset[i], fset[j], nu, lambda);
                        if (v < bsf) {
                            idx = j;
                            bsf = v;
                        }

                        REQUIRE(idx_ref == idx);

                        // --- --- --- --- --- --- --- --- --- --- --- ---
                        double v_eap = twe(fset[i], fset[j], nu, lambda, bsf_eap);
                        if (v_eap < bsf_eap) {
                            idx_eap = j;
                            bsf_eap = v_eap;
                        }

                        REQUIRE(idx_ref == idx_eap);
                    }
                }// End query loop
            }
        }
    }// End section
}
