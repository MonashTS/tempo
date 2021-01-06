#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <tempo/tseries/tseries.hpp>
#include <tempo/univariate/classifiers/nn1/nn1.hpp>
#include <tempo/univariate/distances/dtw/cdtw.hpp>
#include <tempo/univariate/distances/dtw/dtw.hpp>

#include "../tests_tools.hpp"

using namespace tempo::univariate;
constexpr double POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<double>;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("NN1 CDTW Fixed length") {

    // Create 2 random datasets of TSeries
    constexpr int nbitems = ttools::def_nbitems;
    constexpr int fixed = ttools::def_fixed_size;

    std::vector<tempo::TSeries<double, int>> train;
    train.reserve(nbitems);
    for(auto&& s: ttools::get_set_fixed_length(ttools::prng, nbitems, fixed)){
        train.emplace_back(std::move(s), 1, false, std::optional<int>(0));
    }

    std::vector<tempo::TSeries<double, int>> test;
    test.reserve(nbitems);
    for(auto&& s: ttools::get_set_fixed_length(ttools::prng, nbitems, fixed)){
        test.emplace_back(std::move(s), 1, false, std::optional<int>());
    }

    // --- --- --- DTW
    {
        distfun_t<double, int> f = tempo::univariate::distfun_dtw<double, int>();
        distfun_cutoff_t<double, int> fco = tempo::univariate::distfun_cutoff_dtw<double, int>();

        for (const auto &q: test) {
            auto res = nn1<double, int>(f, fco, train.begin(), train.end(), q);
            REQUIRE(res.size() == 1);
        }
    }

    // --- --- --- CDTW
    {
        distfun_t<double, int> f = tempo::univariate::distfun_cdtw<double, int>(2);
        distfun_cutoff_t<double, int> fco = tempo::univariate::distfun_cutoff_cdtw<double, int>(2);

        for (const auto &q: test) {
            auto res = nn1<double, int>(f, fco, train.begin(), train.end(), q);
            REQUIRE(res.size() == 1);
        }
    }

}
