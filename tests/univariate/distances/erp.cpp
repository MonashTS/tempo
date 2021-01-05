#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <tempo/univariate/distances/erp/erp.hpp>

#include "../tests_tools.hpp"
#include "references/erp/erp.hpp"

using namespace tempo::univariate;
constexpr double POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<double>;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

TEST_CASE("ERP Fixed length", "[erp]") {

    // Create a random dataset
    constexpr int nbitems = ttools::def_nbitems;
    constexpr int fsize = ttools::def_fixed_size;
    const auto fset = ttools::get_set_fixed_length(ttools::prng, nbitems, fsize);

    SECTION("ERP(s,s) == 0") {
        for(auto wr: ttools::wratios) {
            size_t w = wr*fsize;
            for (auto gv: ttools::gvalues) {
                for (const auto &series: fset) {
                    double vref = reference::erp_matrix(series, series, gv, w);
                    REQUIRE(vref == 0);

                    double v = erp(series, series, gv, w);
                    REQUIRE(v == 0);

                    double v_eap = erp(series, series, gv, w, POSITIVE_INFINITY);
                    REQUIRE(v_eap == 0);
                }
            }
        }
    }

    SECTION("ERP(s1, s2)") {
        for(auto wr: ttools::wratios) {
            size_t w = wr * fsize;
            for (auto gv: ttools::gvalues) {
                for (int i = 0; i < nbitems; i += 2) {
                    const auto &series1 = fset[i];
                    const auto &series2 = fset[i + 1];

                    double vref = reference::erp_matrix(series1, series2, gv, w);
                    INFO("Exact same operation order. Expect exact floating point equality.")

                    double v = erp(series1, series2, gv, w);
                    REQUIRE(vref == v);

                    double v_eap = erp(series1, series2, gv, w, POSITIVE_INFINITY);
                    REQUIRE(vref == v_eap);
                }
            }
        }
    }

    SECTION("NN1 ERP"){

        for(auto wr: ttools::wratios) {
            size_t w = wr * fsize;

            for (auto gv: ttools::gvalues) {
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
                        double v_ref = reference::erp_matrix(fset[i], fset[j], gv, w);
                        if (v_ref < bsf_ref) {
                            idx_ref = j;
                            bsf_ref = v_ref;
                        }

                        // --- --- --- --- --- --- --- --- --- --- --- ---
                        double v = erp(fset[i], fset[j], gv, w);
                        if (v < bsf) {
                            idx = j;
                            bsf = v;
                        }

                        REQUIRE(idx_ref == idx);

                        // --- --- --- --- --- --- --- --- --- --- --- ---
                        double v_eap = erp(fset[i], fset[j], gv, w, bsf_eap);
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


TEST_CASE("ERP variable length", "[erp]") {

    // Create a random dataset
    constexpr int nbitems = ttools::def_nbitems;
    const auto fset = ttools::get_set_variable_length(ttools::prng, nbitems);

    SECTION("ERP(s,s) == 0") {
        for(auto wr: ttools::wratios) {
            for (auto gv: ttools::gvalues) {
                for (const auto &series: fset) {
                    size_t w = wr*series.size();

                    double vref = reference::erp_matrix(series, series, gv, w);
                    REQUIRE(vref == 0);

                    double v = erp(series, series, gv, w);
                    REQUIRE(v == 0);

                    double v_eap = erp(series, series, gv, w, POSITIVE_INFINITY);
                    REQUIRE(v_eap == 0);
                }
            }
        }
    }

    SECTION("ERP(s1, s2)") {
        for(auto wr: ttools::wratios) {
            for (auto gv: ttools::gvalues) {
                for (int i = 0; i < nbitems; i += 2) {
                    const auto &series1 = fset[i];
                    const auto &series2 = fset[i + 1];
                    size_t w = wr*series1.size();

                    double vref = reference::erp_matrix(series1, series2, gv, w);
                    INFO("Exact same operation order. Expect exact floating point equality.")
                    INFO("size i,j = " << series1.size() << " " << series2.size() << " w= " << w);

                    double v = erp(series1, series2, gv, w);
                    REQUIRE(vref == v);

                    double v_eap = erp(series1, series2, gv, w, POSITIVE_INFINITY);
                    REQUIRE(vref == v_eap);
                }
            }
        }
    }

    SECTION("NN1 ERP"){

        for(auto wr: ttools::wratios) {
            for (auto gv: ttools::gvalues) {
                // Query loop
                for (int i = 0; i < nbitems; i += 3) {
                    const size_t w = wr*fset[i].size();

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
                        double v_ref = reference::erp_matrix(fset[i], fset[j], gv, w);
                        if (v_ref < bsf_ref) {
                            idx_ref = j;
                            bsf_ref = v_ref;
                        }

                        // --- --- --- --- --- --- --- --- --- --- --- ---
                        double v = erp(fset[i], fset[j], gv, w);
                        if (v < bsf) {
                            idx = j;
                            bsf = v;
                        }

                        REQUIRE(idx_ref == idx);

                        // --- --- --- --- --- --- --- --- --- --- --- ---
                        double v_eap = erp(fset[i], fset[j], gv, w, bsf_eap);
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
