#define CATCH_CONFIG_FAST_COMPILE

#include <iostream>

#include <catch.hpp>
#include <tempo/tseries/tseries.hpp>
#include <tempo/tseries/dataset.hpp>

#include "../univariate/tests_tools.hpp"


// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Univariate Dataset") {

    using namespace tempo;

    // Create a random datasets of TSeries
    constexpr int nbitems = ttools::def_nbitems;
    constexpr int fixed = ttools::def_fixed_size;
    std::vector<TSeries<double, int>> dataset_v_;
    dataset_v_.reserve(nbitems);
    for(auto&& s: ttools::get_set_fixed_length(ttools::prng, nbitems, fixed)){
        dataset_v_.emplace_back(std::move(s), 1, false, std::optional<int>(0));
    }
    auto dataset_shd = std::make_shared<std::vector<TSeries<double, int>>>(std::move(dataset_v_));

    Dataset<double, int> dataset(dataset_shd);

    // Test the created dataset
    REQUIRE(dataset.size() == nbitems);
    REQUIRE(dataset.store_info().min_length == fixed);
    REQUIRE(dataset.store_info().max_length == fixed);
    REQUIRE(dataset.store_info().has_missing == false);
    REQUIRE(dataset.store_info().nb_dimensions == 1);
    REQUIRE(dataset.store_info().size == nbitems);
    REQUIRE(dataset.store_info().labels == std::set<int>{0});
    {
        size_t idx = 0;
        for (const auto &ts: dataset) {
            REQUIRE(ts == dataset_shd->at(idx));
            ++idx;
        }
    }

    // Range Subsets
    auto range0 = Dataset(dataset, 0, nbitems);
    REQUIRE(range0.size() == dataset.size());
    {
        size_t idx = 0;
        for (const auto &ts:range0) {
            REQUIRE(ts == dataset_shd->at(idx));
            ++idx;
        }
    }

    // Set Subsets
    std::vector<size_t> s0;
    for(size_t i=0; i<nbitems; ++i){s0.push_back(i);}
    auto set0 = Dataset(dataset, s0);
    REQUIRE(set0.size() == dataset.size());
    {
        size_t idx = 0;
        for (const auto &ts:set0) {
            REQUIRE(ts == dataset_shd->at(idx));
            ++idx;
        }
    }

    // Quick gini test
    REQUIRE(0 == gini_impurity(getByClassMap(dataset)));

}
