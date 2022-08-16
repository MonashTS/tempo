#include <catch2/catch_test_macros.hpp>

#include "direct.hpp"

#include <mock/mockseries.hpp>

#include <vector>

using namespace tempo::distance;

using F = double;

constexpr size_t nbitems = 500;
constexpr auto cfun = tempo::distance::univariate::idx_ad2<F, std::vector<F>>;
constexpr F PINF = utils::PINF<F>;

namespace ref {

  template<typename V>
  double directa(const V& s1, const V& s2) {
    if (s1.size()!=s2.size()) { return PINF; }
    double cost = 0;
    for (size_t i = 0; i<s1.size(); ++i) {
      const auto d = s1[i] - s2[i];
      cost += d*d;
    }
    return cost;
  }

}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Univariate NORM Fixed length", "[directa][univariate]") {
  // Setup univariate with fixed length
  mock::Mocker mocker(0);

  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("NORM(s,s) == 0") {
    for (const auto& s : fset) {
      const double directa_ref_v = ref::directa(s, s);
      REQUIRE(directa_ref_v==0);

      const auto directa_v = directa(s.size(), s.size(), cfun(s, s), PINF);
      REQUIRE(directa_v==0);
    }
  }

  SECTION("NORM(s1, s2)") {
    for (size_t i = 0; i<nbitems - 1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i + 1];

      const double directa_ref_v = ref::directa(s1, s2);
      INFO("Exact same operation order. Expect exact floating point equality.");

      const auto directa_tempo_v = directa(s1.size(), s2.size(), cfun(s1, s2), PINF);
      REQUIRE(directa_ref_v==directa_tempo_v);
    }
  }

  SECTION("NN1 NORM") {
    // Query loop
    for (size_t i = 0; i<nbitems; i += 3) {
      const auto& s1 = fset[i];
      // Ref Variables
      size_t idx_ref = 0;
      double bsf_ref = PINF;
      // Base Variables
      size_t idx = 0;
      double bsf = PINF;
      // EAP Variables
      size_t idx_tempo = 0;
      double bsf_tempo = PINF;

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const double v_ref = ref::directa(s1, s2);
        if (v_ref<bsf_ref) {
          idx_ref = j;
          bsf_ref = v_ref;
        }

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto v = directa(s1.size(), s2.size(), cfun(s1, s2), PINF);
        if (v<bsf) {
          idx = j;
          bsf = v;
        }

        REQUIRE(idx_ref==idx);

        // --- --- --- --- --- --- --- --- --- --- --- ---
        const auto v_tempo = directa(s1.size(), s2.size(), cfun(s1, s2), bsf_tempo);
        if (v_tempo<bsf_tempo) {
          idx_tempo = j;
          bsf_tempo = v_tempo;
        }

        INFO(i << " "  << j << " vtempo = " << v_tempo << "  bsf_tempo = " << bsf_tempo << "    v = "  << v );
        REQUIRE(idx_ref==idx_tempo);
      }
    }// End query loop
  }// End section
}
