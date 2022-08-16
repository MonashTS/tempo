#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "wdtw.hpp"

#include <mock/mockseries.hpp>

#include <vector>

using namespace tempo::distance;

using F = double;

constexpr size_t nbitems = 500;
constexpr size_t nbweights = 5;
constexpr auto cfun = univariate::idx_ad2<F, std::vector<F>>;
constexpr F PINF = utils::PINF<F>;

F wdtw(std::vector<F> const& lines, std::vector<F> const& cols, std::vector<F> const& weights, F cutoff) {
  return wdtw(lines.size(), cols.size(), cfun(lines, cols), weights, cutoff);
}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace ref {

  /** Reimplementation of the "modified logistic weight function (MLWF)" from the original paper
   * "Weighted dynamic time warping for time series classification"
   * @param i Index of the point in [1..m] (m=length of the sequence)
   * @param mc Mid point of the sequence (m/2)
   * @param g "Controls the level of penalization for the points with larger phase difference".
   *        range [0, +inf), usually in [0.01, 0.6].
   *        Some examples:
   *        * 0: constant weight
   *        * 0.05: nearly linear weights
   *        * 0.25: sigmoid weights
   *        * 3: two distinct weights between half sequences
   * @param wmax Upper bound for the weight parameter. Keep it to 1
   * @return
   */
  inline double mlwf(double i, double mc, double g, double wmax = 1) {
    return wmax/(1 + std::exp(-g*(i - mc)));
  }

  /// Reference implementation on a matrix
  double wdtw_matrix(
    std::vector<double> const& series1,
    std::vector<double> const& series2,
    std::vector<double> const& weights
  ) {
    const long length1 = utils::to_signed(series1.size());
    const long length2 = utils::to_signed(series2.size());

    // Check lengths. Be explicit in the conditions.
    if (length1==0&&length2==0) { return 0; }
    if (length1==0&&length2!=0) { return PINF; }
    if (length1!=0&&length2==0) { return PINF; }

    constexpr auto sqdist = univariate::ad2<F>;

    // Matrix
    std::vector<std::vector<double>> matrix(length1, std::vector<double>(length2, 0));
    // First value
    matrix[0][0] = weights[0]*sqdist(series1[0], series2[0]);
    // First line
    for (long i = 1; i<length2; i++) {
      matrix[0][i] = matrix[0][i - 1] + weights[i]*sqdist(series1[0], series2[i]);
    }
    // First column
    for (long i = 1; i<length1; i++) {
      matrix[i][0] = matrix[i - 1][0] + weights[i]*sqdist(series1[i], series2[0]);
    }
    // Matrix computation
    for (long i = 1; i<length1; i++) {
      for (long j = 1; j<length2; j++) {
        const auto d = weights[std::abs(i - j)]*sqdist(series1[i], series2[j]);
        const auto v = utils::min(matrix[i][j - 1], matrix[i - 1][j], matrix[i - 1][j - 1]) + d;
        matrix[i][j] = v;
      }
    }
    return matrix[length1 - 1][length2 - 1];
  }

}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

TEST_CASE("Test weights generation", "[wdtw]") {
  // Random weight factors
  mock::Mocker mocker;
  const auto weight_factors = mocker.randvec(nbweights, 0, 1);
  constexpr size_t maxlength = 50;
  constexpr double mc = ((double)maxlength)/2;
  for (double g : weight_factors) {
    auto weights = wdtw_weights(g, maxlength);
    for (size_t i = 0; i<maxlength; ++i) {
      auto lib = weights[i];
      auto ref = ref::mlwf((double)i, mc, g);
      REQUIRE(lib==ref);
    }
  }
}

TEST_CASE("Univariate WDTW Fixed length", "[wdtw][univariate]") {
  // Setup univariate with fixed length
  mock::Mocker mocker;
  const auto fset = mocker.vec_randvec(nbitems);
  // Random weight factors
  const auto weight_factors = mocker.randvec(nbweights, 0, 1);

  SECTION("WDTW(s,s) == 0") {
    for (const auto& s : fset) {
      for (double g : weight_factors) {
        auto weights = wdtw_weights(g, mocker._fixl);

        const double dtw_ref_v = ref::wdtw_matrix(s, s, weights);
        REQUIRE(dtw_ref_v==0);

        const auto dtw_v = wdtw(s, s, weights, PINF);
        REQUIRE(dtw_v==0);
      }
    }
  }

  SECTION("WDTW(s1, s2)") {
    for (size_t i = 0; i<nbitems - 1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i + 1];

      for (double g : weight_factors) {
        auto weights = wdtw_weights(g, mocker._fixl);

        const double dtw_ref_v = ref::wdtw_matrix(s1, s2, weights);
        INFO("Exact same operation order. Expect exact floating point equality.");

        const auto dtw_tempo = wdtw(s1, s2, weights, PINF);
        REQUIRE(dtw_ref_v==dtw_tempo);
      }
    }
  }

  SECTION("NN1 WDTW") {
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
        // Create the univariate squared Euclidean distance for our dtw functions
        for (double g : weight_factors) {
          auto weights = wdtw_weights(g, mocker._fixl);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = ref::wdtw_matrix(s1, s2, weights);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = wdtw(s1, s2, weights, PINF);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_tempo = wdtw(s1, s2, weights, bsf_tempo);
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

TEST_CASE("Univariate WDTW Variable length", "[wdtw][univariate]") {
  // Setup univariate with fixed length
  mock::Mocker mocker;
  const auto fset = mocker.vec_rs_randvec(nbitems);
  // Random weight factors
  const auto weight_factors = mocker.randvec(nbweights, 0, 1);

  SECTION("WDTW(s,s) == 0") {
    for (const auto& s : fset) {
      for (double g : weight_factors) {
        auto weights = wdtw_weights(g, s.size());
        const double dtw_ref_v = ref::wdtw_matrix(s, s, weights);
        REQUIRE(dtw_ref_v==0);

        const auto dtw_v = wdtw(s, s, weights, PINF);
        REQUIRE(dtw_v==0);
      }
    }
  }

  SECTION("WDTW(s1, s2)") {
    for (size_t i = 0; i<nbitems - 1; ++i) {
      for (double g : weight_factors) {
        const auto& s1 = fset[i];
        const auto& s2 = fset[i + 1];
        auto weights = wdtw_weights(g, (std::max(s1.size(), s2.size())));

        const double dtw_ref_v = ref::wdtw_matrix(s1, s2, weights);
        INFO("Exact same operation order. Expect exact floating point equality.");

        const auto dtw_tempo_v = wdtw(s1, s2, weights, PINF);
        REQUIRE(dtw_ref_v==dtw_tempo_v);
      }
    }
  }

  SECTION("NN1 WDTW") {
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
        // Create the univariate squared Euclidean distance for our dtw functions

        for (double g : weight_factors) {
          auto weights = wdtw_weights(g, (std::max(s1.size(), s2.size())));

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = ref::wdtw_matrix(s1, s2, weights);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = wdtw(s1, s2, weights, PINF);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }

          REQUIRE(idx_ref==idx);

          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_tempo = wdtw(s1, s2, weights, bsf_tempo);
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
