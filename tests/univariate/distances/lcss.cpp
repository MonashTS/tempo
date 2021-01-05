#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <tempo/univariate/distances/lcss/lcss.hpp>

#include "../tests_tools.hpp"
#include "references/lcss/lcss.hpp"

using namespace tempo::univariate;
constexpr double POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<double>;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

TEST_CASE("LCSS Fixed length", "[lcss]") {

    // Create a random dataset
    constexpr int nbitems = ttools::def_nbitems;
    constexpr int fsize = ttools::def_fixed_size;
    const auto fset = ttools::get_set_fixed_length(ttools::prng, nbitems, fsize);

    SECTION("LCSS(s,s) == 0") {
        for(auto wr: ttools::wratios) {
            size_t w = wr*fsize;
            for (auto epsilon: ttools::epsilons) {
                for (const auto &series: fset) {

                    double vref = reference::lcss_matrix(series, series, epsilon, w);
                    REQUIRE(vref == 0);

                    double v = lcss(series, series, epsilon, w);
                    REQUIRE(v == 0);

                    double vea = lcss(series, series, epsilon, w, POSITIVE_INFINITY);
                    REQUIRE(vea == 0);
                }
            }
        }
    }

    SECTION("LCSS(s1, s2)") {
        for(auto wr: ttools::wratios) {
            size_t w = wr * fsize;
            for (auto epsilon: ttools::epsilons) {
                for (int i = 0; i < nbitems; i += 2) {
                    const auto &series1 = fset[i];
                    const auto &series2 = fset[i + 1];

                    double vref = reference::lcss_matrix(series1, series2, epsilon, w);
                    INFO("Exact same operation order. Expect exact floating point equality.")

                    double v = lcss(series1, series2, epsilon, w);
                    REQUIRE(vref == v);

                    double vea = lcss(series1, series2, epsilon, w, POSITIVE_INFINITY);
                    REQUIRE(vref == vea);
                }
            }
        }
    }

    SECTION("NN1 LCSS"){
        for(auto wr: ttools::wratios) {

            size_t w = wr * fsize;
            for (auto epsilon:ttools::epsilons) {

                // Query loop
                for (int i = 0; i < nbitems; i += 3) {
                    // Ref Variables
                    int idx_ref = 0;
                    double bsf_ref = POSITIVE_INFINITY;

                    // Base Variables
                    int idx = 0;
                    double bsf = POSITIVE_INFINITY;

                    // EA Variables
                    int idx_ea = 0;
                    double bsf_ea = POSITIVE_INFINITY;

                    // NN1 loop
                    for (int j = 0; j < nbitems; j += 5) {
                        // Skip self.
                        if (i == j) { continue; }

                        // --- --- --- --- --- --- --- --- --- --- --- ---
                        double v_ref = reference::lcss_matrix(fset[i], fset[j], epsilon, w);
                        if (v_ref < bsf_ref) {
                            idx_ref = j;
                            bsf_ref = v_ref;
                        }

                        // --- --- --- --- --- --- --- --- --- --- --- ---
                        double v = lcss(fset[i], fset[j], epsilon, w);
                        if (v < bsf) {
                            idx = j;
                            bsf = v;
                        }

                        REQUIRE(idx_ref == idx);

                        // --- --- --- --- --- --- --- --- --- --- --- ---
                        double v_ea = lcss(fset[i], fset[j], epsilon, w, bsf_ea);
                        if (v_ea < bsf_ea) {
                            idx_ea = j;
                            bsf_ea = v_ea;
                        }

                        REQUIRE(idx_ref == idx_ea);

                    }
                }// End query loop
            }
        }
    }// End section
}


TEST_CASE("LCSS variable length", "[lcss]") {

    // Create a random dataset
    constexpr int nbitems = ttools::def_nbitems;
    const auto fset = ttools::get_set_variable_length(ttools::prng, nbitems);

    SECTION("LCSS(s,s) == 0") {
        for(auto wr: ttools::wratios) {
            for (auto epsilon: ttools::epsilons) {
                for (const auto &series: fset) {
                    size_t w = wr*series.size();

                    double vref = reference::lcss_matrix(series, series, epsilon, w);
                    REQUIRE(vref == 0);

                    double v = lcss(series, series, epsilon, w);
                    REQUIRE(v == 0);

                    double vea = lcss(series, series, epsilon, w, POSITIVE_INFINITY);
                    REQUIRE(vea == 0);
                }
            }
        }
    }

    SECTION("LCSS(s1, s2)") {
        for(auto wr: ttools::wratios) {
            for (auto epsilon: ttools::epsilons) {
                for (int i = 0; i < nbitems; i += 2) {
                    const auto &series1 = fset[i];
                    const auto &series2 = fset[i + 1];
                    size_t w = wr*series1.size();

                    double vref = reference::lcss_matrix(series1, series2, epsilon, w);
                    INFO("Exact same operation order. Expect exact floating point equality.")

                    double v = lcss(series1, series2, epsilon, w);
                    REQUIRE(vref == v);

                    double vea = lcss(series1, series2, epsilon, w, POSITIVE_INFINITY);
                    REQUIRE(vref == vea);
                }
            }
        }
    }

    SECTION("NN1 LCSS"){
        for(auto wr: ttools::wratios) {
            for (auto epsilon: ttools::epsilons) {

                // Query loop
                for (int i = 0; i < nbitems; i += 3) {
                    const size_t w = wr*fset[i].size();

                    // Ref Variables
                    int idx_ref = 0;
                    double bsf_ref = POSITIVE_INFINITY;

                    // Base Variables
                    int idx = 0;
                    double bsf = POSITIVE_INFINITY;

                    // EA Variables
                    int idx_ea = 0;
                    double bsf_ea = POSITIVE_INFINITY;

                    // NN1 loop
                    for (int j = 0; j < nbitems; j += 5) {
                        // Skip self.
                        if (i == j) { continue; }

                        // --- --- --- --- --- --- --- --- --- --- --- ---
                        double v_ref = reference::lcss_matrix(fset[i], fset[j], epsilon, w);
                        if (v_ref < bsf_ref) {
                            idx_ref = j;
                            bsf_ref = v_ref;
                        }

                        // --- --- --- --- --- --- --- --- --- --- --- ---
                        double v = lcss(fset[i], fset[j], epsilon, w);
                        if (v < bsf) {
                            idx = j;
                            bsf = v;
                        }

                        REQUIRE(idx_ref == idx);

                        // --- --- --- --- --- --- --- --- --- --- --- ---
                        double v_ea = lcss(fset[i], fset[j], epsilon, w, bsf_ea);
                        if (v_ea < bsf_ea) {
                            idx_ea = j;
                            bsf_ea = v_ea;
                        }

                        REQUIRE(idx_ref == idx_ea);

                    }
                }// End query loop
            }
        }
    }// End section
}
