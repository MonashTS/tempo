#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <tempo/univariate/distances/dtw/cdtw.hpp>
#include <tempo/univariate/distances/dtw/lowerbound/lb_keogh.hpp>
#include <tempo/univariate/distances/dtw/lowerbound/lb_enhanced.hpp>
#include <tempo/univariate/distances/dtw/lowerbound/envelopes.hpp>
#include <iostream>

#include "../tests_tools.hpp"
#include "references/dtw/cdtw.hpp"

using namespace tempo::univariate;
constexpr double POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<double>;


// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("CDTW Fixed length", "[cdtw]") {
    Catch::StringMaker<double>::precision = 18;

    // Create a random dataset
    constexpr int nbitems = ttools::def_nbitems;
    constexpr int fixed = ttools::def_fixed_size;
    const auto fset = ttools::get_set_fixed_length(ttools::prng, nbitems, fixed);

    SECTION("CDTW(s,s) == 0") {
        for (const auto &series: fset) {

            for(const auto wr: ttools::wratios) {
                size_t w = wr * fixed;

                const double cdtw_ref_v = reference::cdtw_matrix(series, series, w);
                REQUIRE(cdtw_ref_v == 0);

                const double cdtw_v = cdtw(series, series, w);
                REQUIRE(cdtw_v == 0);

                const double cdtw_eap_v = cdtw(series, series, w, POSITIVE_INFINITY);
                REQUIRE(cdtw_eap_v == 0);
            }
        }
    }

    SECTION("CDTW(s1, s2)") {
        for (int i = 0; i < nbitems; i += 2) {
            const auto &series1 = fset[i];
            const auto &series2 = fset[i + 1];

            for(const auto wr: ttools::wratios) {
                size_t w = wr * fixed;

                const double cdtw_ref_v = reference::cdtw_matrix(series1, series2, w);
                INFO("Exact same operation order. Expect exact floating point equality.")

                const double cdtw_v = cdtw(series1, series2, w);
                REQUIRE(cdtw_ref_v == cdtw_v);

                const double cdtw_eap_v = cdtw(series1, series2, w, POSITIVE_INFINITY);
                REQUIRE(cdtw_ref_v == cdtw_eap_v);
            }
        }
    }

    SECTION("NN1 CDTW"){

        // Query loop
        for (int i = 0; i < nbitems; i += 3) {

            // Window loop
            for(const auto wr: ttools::wratios) {
                size_t w = wr * fixed;

                // Ref Variables
                int idx_ref = 0;
                double bsf_ref = POSITIVE_INFINITY;

                // Base Variables
                int idx = 0;
                double bsf = POSITIVE_INFINITY;

                // EAP Variables
                int idx_eap = 0;
                double bsf_eap = POSITIVE_INFINITY;

                // LB KEOGH
                int idx_lbk = 0;
                double bsf_lbk = POSITIVE_INFINITY;
                const auto& candidate = fset[i];
                std::vector<double> up, lo;
                get_keogh_envelopes(candidate, up, lo, w);

                // LB Enhanced
                int idx_lbe = 0;
                double bsf_lbe = POSITIVE_INFINITY;
                // Reuse lb keogh envelope
                // std::vector<double> up, lo;
                // get_keogh_envelopes(candidate, up, lo, w);

                // NN1 loop
                for (int j = 0; j < nbitems; j += 5) {
                    // Skip self.
                    if (i == j) { continue; }
                    const auto& query = fset[j];

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    double v_ref = reference::cdtw_matrix(fset[i], fset[j], w);
                    if (v_ref < bsf_ref) {
                        idx_ref = j;
                        bsf_ref = v_ref;
                    }

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    double v = cdtw(fset[i], fset[j], w);
                    if (v < bsf) {
                        idx = j;
                        bsf = v;
                    }

                    REQUIRE(idx_ref == idx);

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    double v_eap = cdtw(fset[i], fset[j], w, bsf_eap);
                    if (v_eap < bsf_eap) {
                        idx_eap = j;
                        bsf_eap = v_eap;
                    }

                    REQUIRE(idx_ref == idx_eap);

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    // Test lb keogh: must be a lower bound
                    double vk = lb_Keogh(query.data(), query.size(), up.data(), lo.data(), candidate.size(), w, bsf_lbk);
                    if(vk!=POSITIVE_INFINITY){ REQUIRE(vk<=v_eap); }
                    if(vk<=bsf_lbk){
                        double v_lbk = cdtw(candidate, query, w, bsf_lbk);
                        if(v_lbk < bsf_lbk){
                            idx_lbk = j;
                            bsf_lbk = v_lbk;
                        }
                    }

                    REQUIRE(idx_ref == idx_lbk);

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    // Test lb enhanced: must be a lower bound
                    double ve = lb_Enhanced(query, candidate, up, lo, 3, w, bsf_lbe);
                    INFO( query.size() << " " << candidate.size() << " cutoff = " << bsf_lbe << " w = " << w);
                    if(ve!=POSITIVE_INFINITY){ REQUIRE(ve<=v_eap); }
                    if(ve<=bsf_lbe){
                        double v_lbe = cdtw(candidate, query, w, bsf_lbe);
                        if(v_lbe < bsf_lbe){
                            idx_lbe = j;
                            bsf_lbe = v_lbe;
                        }
                    }

                    REQUIRE(idx_ref == idx_lbe);
                }
            } // End window loop
        }// End query loop
    }// End section
}


TEST_CASE("CDTW variable length", "[cdtw]") {
    Catch::StringMaker<double>::precision = 18;

    // Create a random dataset
    constexpr int nbitems = ttools::def_nbitems;
    const auto fset = ttools::get_set_variable_length(ttools::prng, nbitems);

    SECTION("CDTW(s,s) == 0") {
        for (const auto &series: fset) {

            for(const auto wr: ttools::wratios) {
                size_t w = wr * series.size();

                const double cdtw_ref_v = reference::cdtw_matrix(series, series, w);
                REQUIRE(cdtw_ref_v == 0);

                const double cdtw_v = cdtw(series, series, w);
                REQUIRE(cdtw_v == 0);

                const double cdtw_eap_v = cdtw(series, series, w, POSITIVE_INFINITY);
                REQUIRE(cdtw_eap_v == 0);
            }
        }
    }

    SECTION("CDTW(s1, s2)") {
        for (int i = 0; i < nbitems; i += 2) {
            const auto& series1 = fset[i];
            const auto& series2 = fset[i + 1];

            for(const auto wr: ttools::wratios) {
                size_t w = wr * series1.size();

                const double cdtw_ref_v = reference::cdtw_matrix(series1, series2, w);
                INFO("Exact same operation order. Expect exact floating point equality.")

                const double cdtw_v = cdtw(series1, series2, w);
                REQUIRE(cdtw_ref_v == cdtw_v);

                const double cdtw_eap_v = cdtw(series1, series2, w, POSITIVE_INFINITY);
                REQUIRE(cdtw_ref_v == cdtw_eap_v);
            }
        }
    }

    SECTION("NN1 CDTW"){
        // Candidate loop
        for(int i=0; i<nbitems; i+=3) {
            // Windows loop
            for(const auto wr: ttools::wratios) {
                // Compute window
                size_t w = wr * fset[i].size();

                // Ref Variables
                int idx_ref=0;
                double bsf_ref = POSITIVE_INFINITY;

                // Base Variables
                int idx=0;
                double bsf = POSITIVE_INFINITY;

                // EAP Variables
                int idx_eap = 0;
                double bsf_eap = POSITIVE_INFINITY;

                // LB KEOGH
                int idx_lbk = 0;
                double bsf_lbk = POSITIVE_INFINITY;
                const auto& candidate = fset[i];
                std::vector<double> up, lo;
                get_keogh_envelopes(candidate, up, lo, w);

                // NN1 loop
                for (int j = 0; j < nbitems; j+=5) {
                    // Skip self.
                    if(i==j){continue;}
                    const auto& query = fset[j];

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    double v_ref = reference::cdtw_matrix(fset[i], fset[j], w);
                    if (v_ref < bsf_ref) {
                        idx_ref = j;
                        bsf_ref = v_ref;
                    }

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    double v = cdtw(fset[i], fset[j], w);
                    if (v < bsf) {
                        idx = j;
                        bsf = v;
                    }
                    REQUIRE(idx_ref == idx);

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    double v_eap = cdtw(fset[i], fset[j], w, bsf_eap);
                    if (v_eap < bsf_eap) {
                        idx_eap = j;
                        bsf_eap = v_eap;
                    }

                    REQUIRE(idx_ref == idx_eap);

                    // --- --- --- --- --- --- --- --- --- --- --- ---
                    // Test lb keogh: must be a lower bound
                    double vk = lb_Keogh(query.data(), query.size(), up.data(), lo.data(), candidate.size(), w, bsf_lbk);
                    if(vk!=POSITIVE_INFINITY){ REQUIRE(vk<=v_eap); }
                    if(vk<=bsf_lbk){
                        double v_lbk = cdtw(candidate, query, w, bsf_lbk);
                        if(v_lbk < bsf_lbk){
                            idx_lbk = j;
                            bsf_lbk = v_lbk;
                        }
                    }

                    REQUIRE(idx_ref == idx_lbk);
                }
            }// End window loop
        }// End candidate loop
    }// End Section
}
