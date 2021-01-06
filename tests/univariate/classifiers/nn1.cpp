#define CATCH_CONFIG_FAST_COMPILE

#include <catch.hpp>
#include <tempo/tseries/tseries.hpp>
#include <tempo/univariate/classifiers/nn1/nn1.hpp>
#include <tempo/univariate/distances/dtw/cdtw.hpp>
#include <tempo/univariate/distances/dtw/dtw.hpp>
#include <tempo/univariate/distances/dtw/wdtw.hpp>
#include <tempo/univariate/distances/elementwise/elementwise.hpp>
#include <tempo/univariate/distances/erp/erp.hpp>
#include <tempo/univariate/distances/lcss/lcss.hpp>
#include <tempo/univariate/distances/msm/msm.hpp>
#include <tempo/univariate/distances/twe/twe.hpp>

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

    // --- --- --- WDTW
    {
        auto weights = std::make_shared<std::vector<double>>(generate_weights(0.1, fixed));
        auto f = tempo::univariate::distfun_wdtw<double, int>(weights);
        auto fco = tempo::univariate::distfun_cutoff_wdtw<double, int>(weights);

        for (const auto &q: test) {
            auto res = nn1<double, int>(f, fco, train.begin(), train.end(), q);
            REQUIRE(res.size() == 1);
        }
    }

    // --- --- --- EWISE
    {
        auto f = tempo::univariate::distfun_elementwise<double, int>();
        auto fco = tempo::univariate::distfun_cutoff_elementwise<double, int>();

        for (const auto &q: test) {
            auto res = nn1<double, int>(f, fco, train.begin(), train.end(), q);
            REQUIRE(res.size() == 1);
        }
    }

    // --- --- --- ERP
    {
        auto f = tempo::univariate::distfun_erp<double, int>(0.5, 2);
        auto fco = tempo::univariate::distfun_cutoff_erp<double, int>(0.5, 2);

        for (const auto &q: test) {
            auto res = nn1<double, int>(f, fco, train.begin(), train.end(), q);
            REQUIRE(res.size() == 1);
        }
    }

    // --- --- ---  LCSS
    {
        auto f = tempo::univariate::distfun_lcss<double, int>(0.5, 2);
        auto fco = tempo::univariate::distfun_cutoff_lcss<double, int>(0.5, 2);

        for (const auto &q: test) {
            auto res = nn1<double, int>(f, fco, train.begin(), train.end(), q);
            REQUIRE(res.size() == 1);
        }
    }

    // --- --- --- MSM
    {
        auto f = tempo::univariate::distfun_msm<double, int>(0.5);
        auto fco = tempo::univariate::distfun_cutoff_msm<double, int>(0.5);

        for (const auto &q: test) {
            auto res = nn1<double, int>(f, fco, train.begin(), train.end(), q);
            REQUIRE(res.size() == 1);
        }
    }

    // --- --- --- TWE
    {
        auto f = tempo::univariate::distfun_twe<double, int>(0.5, 1);
        auto fco = tempo::univariate::distfun_cutoff_twe<double, int>(0.5, 1);

        for (const auto &q: test) {
            auto res = nn1<double, int>(f, fco, train.begin(), train.end(), q);
            REQUIRE(res.size() == 1);
        }
    }
}
