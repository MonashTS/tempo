#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <tempo/univariate/distances/elementwise/elementwise.hpp>

#include "../tests_tools.hpp"

using namespace tempo::univariate;
constexpr double POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<double>;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


TEST_CASE("SQED Fixed length", "[sqed]") {

    // Create a random dataset
    constexpr int nbitems = ttools::def_nbitems*10;
    constexpr int fixed = ttools::def_fixed_size;
    const auto fset = ttools::get_set_fixed_length(ttools::prng, nbitems, fixed);

    SECTION("SQED(s,s) == 0") {
        for (const auto &series: fset) {
            const double sqed = elementwise(series, series);
            REQUIRE(sqed == 0);

            const double sqed_ea = elementwise(series, series, POSITIVE_INFINITY);
            REQUIRE(sqed_ea == 0);
        }
    }

    SECTION("SQED(s1, s2)") {
        for (int i = 0; i < nbitems; i += 2) {
            const auto &series1 = fset[i];
            const auto &series2 = fset[i + 1];

            const double sqed = elementwise(series1, series2);
            INFO("Different order of operation: approx floating point");

            const double sqed_ea = elementwise(series1, series2, POSITIVE_INFINITY);
            REQUIRE(sqed == sqed_ea);
        }
    }

    SECTION("NN1 SQED"){
        // Query loop
        for(int i=0; i<nbitems; i+=3) {
            // Base Variables
            int idx=0;
            double bsf = POSITIVE_INFINITY;

            // EA Variables
            int idx_ea = 0;
            double bsf_ea = POSITIVE_INFINITY;

            // NN1 loop
            for (int j = 0; j < nbitems; j+=5) {
                // Skip self.
                if(i==j){continue;}

                // --- --- --- --- --- --- --- --- --- --- --- ---
                double v = elementwise(fset[i], fset[j]);
                if (v < bsf) {
                    idx = j;
                    bsf = v;
                }


                // --- --- --- --- --- --- --- --- --- --- --- ---
                double v_ea = elementwise(fset[i], fset[j], bsf_ea);
                if (v_ea < bsf_ea) {
                    idx_ea = j;
                    bsf_ea = v_ea;
                }

                REQUIRE(idx == idx_ea);
            }
        }// End query loop
    }// End section
}
