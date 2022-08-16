#include <catch2/catch_test_macros.hpp>

#include "adtw.hpp"

#include <mock/mockseries.hpp>

#include <vector>

using namespace tempo::distance;

using F = double;

constexpr size_t nbitems = 500;
constexpr auto cfun = tempo::distance::univariate::idx_ad2<F, std::vector<F>>;
constexpr F PINF = utils::PINF<F>;

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace ref {

  /// Naive ADTW with idx_ad2.
  F adtw_matrix(std::vector<F> const& series1, std::vector<F> const& series2, F penalty) {
    const long length1 = utils::to_signed(series1.size());
    const long length2 = utils::to_signed(series2.size());

    // Check lengths. Be explicit in the conditions
    if (length1==0&&length2==0) { return 0; }
    if (length1==0&&length2!=0) { return PINF; }
    if (length1!=0&&length2==0) { return PINF; }

    // Create sqdist
    auto sqdist = cfun(series1, series2);

    // Allocate the working space: full matrix + space for borders (first column / first line)
    size_t msize = std::max(length1, length2) + 1;
    std::vector<std::vector<F>> matrix(msize, std::vector<F>(msize, PINF));

    // Initialisation (all the matrix is initialised at +INF)
    matrix[0][0] = 0;

    // For each line
    // Note: series1 and series2 are 0-indexed while the matrix is 1-indexed (0 being the borders)
    //       hence, we have i-1 and j-1 when accessing series1 and series2
    for (long i = 1; i<=length1; i++) {
      const size_t i1 = i - 1;
      for (long j = 1; j<=length2; j++) {
        F g = sqdist(i1, j - 1);
        F prev = matrix[i][j - 1] + g + penalty;
        F diag = matrix[i - 1][j - 1] + g;
        F top = matrix[i - 1][j] + g + penalty;
        matrix[i][j] = utils::min(prev, diag, top);
      }
    }

    return matrix[length1][length2];
  }

} // End of namespace ref

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Univariate ADTW Fixed length", "[adtw][univariate]") {
  Catch::StringMaker<F>::precision = 18;

  // Setup univariate with fixed length
  mock::Mocker mocker;
  const auto& penalties = mocker.adtw_penalties;
  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("ADTW(s,s) == 0") {
    for (const auto& s : fset) {
      for (F p : penalties) {
        const F adtw_ref_v = ref::adtw_matrix(s, s, p);
        REQUIRE(adtw_ref_v==0);
        const F adtw_v = adtw(s.size(), s.size(), cfun(s, s), p, PINF);
        REQUIRE(adtw_v==0);
      }
    }
  }

  SECTION("ADTW(s1, s2)") {
    for (size_t i = 0; i<nbitems - 1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i + 1];
      for (F p : penalties) {
        const F adtw_ref_v = ref::adtw_matrix(s1, s2, p);
        INFO("Exact same operation order. Expect exact floating point equality.");
        const auto adtw_tempo = adtw(s1.size(), s2.size(), cfun(s1, s2), p, PINF);
        REQUIRE(adtw_ref_v==adtw_tempo);
      }
    }
  }

  SECTION("NN1 ADTW") {
    // Query loop
    for (size_t i = 0; i<nbitems; i += 3) {
      const auto& s1 = fset[i];
      // Ref Variables
      size_t idx_ref = 0;
      F bsf_ref = PINF;
      // Base Variables
      size_t idx = 0;
      F bsf = PINF;
      // EAP Variables
      size_t idx_tempo = 0;
      F bsf_tempo = PINF;

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];
        // Create the univariate squared Euclidean distance for our adtw functions
        for (F p : penalties) {
          // --- --- --- --- --- --- --- --- --- --- --- ---
          const F v_ref = ref::adtw_matrix(s1, s2, p);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = adtw(s1.size(), s2.size(), cfun(s1, s2), p, PINF);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_tempo = adtw(s1.size(), s2.size(), cfun(s1, s2), p, bsf_tempo);
          if (v_tempo<bsf_tempo) {
            idx_tempo = j;
            bsf_tempo = v_tempo;
          }

          REQUIRE(idx_ref==idx_tempo);
        }
      }
    }// End query loop
  }// End section

}

TEST_CASE("Univariate ADTW Variable length", "[adtw][univariate]") {
  // Setup univariate dataset with varying length
  mock::Mocker mocker(1);
  const auto& penalties = mocker.adtw_penalties;
  const auto fset = mocker.vec_rs_randvec(nbitems);

  SECTION("ADTW(s,s) == 0") {
    for (const auto& s : fset) {
      for (F p : penalties) {
        const F adtw_ref_v = ref::adtw_matrix(s, s, p);
        REQUIRE(adtw_ref_v==0);
        const auto adtw_v = adtw(s.size(), s.size(), cfun(s, s), p, PINF);
        REQUIRE(adtw_v==0);
      }
    }
  }

  SECTION("ADTW(s1, s2)") {
    for (size_t i = 28; i<nbitems - 1; ++i) {
      for (size_t pi = 44; pi<penalties.size(); ++pi) {
        const auto p = penalties[pi];
        const auto& s1 = fset[i];
        const auto& s2 = fset[i + 1];
        const F adtw_ref_v = ref::adtw_matrix(s1, s2, p);
        INFO("Exact same operation order. Expect exact floating point equality.");
        const auto adtw_tempo_v = adtw(s1.size(), s2.size(), cfun(s1, s2), p, PINF);
        INFO(i << " " << pi << " length s1 = " << s1.size() << "  length s2 = " << s2.size());
        REQUIRE(adtw_ref_v==adtw_tempo_v);
      }
    }
  }

  SECTION("NN1 ADTW") {
    // Query loop
    for (size_t i = 0; i<nbitems; i += 3) {
      const auto& s1 = fset[i];
      // Ref Variables
      size_t idx_ref = 0;
      F bsf_ref = PINF;
      // Base Variables
      size_t idx = 0;
      F bsf = PINF;
      // EAP Variables
      size_t idx_tempo = 0;
      F bsf_tempo = PINF;

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];
        // Create the univariate squared Euclidean distance for our adtw functions

        for (F p : penalties) {
          // --- --- --- --- --- --- --- --- --- --- --- ---
          const F v_ref = ref::adtw_matrix(s1, s2, p);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = adtw(s1.size(), s2.size(), cfun(s1, s2), p, PINF);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_tempo = adtw(s1.size(), s2.size(), cfun(s1, s2), p, bsf_tempo);
          if (v_tempo<bsf_tempo) {
            idx_tempo = j;
            bsf_tempo = v_tempo;
          }

          REQUIRE(idx_ref==idx_tempo);
        }
      }
    }// End query loop
  }// End section
}
