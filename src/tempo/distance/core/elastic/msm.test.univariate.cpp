#include <catch2/catch_test_macros.hpp>

#include "msm.hpp"

#include <mock/mockseries.hpp>

#include <vector>

using namespace tempo::distance;

using F = double;

constexpr size_t nbitems = 500;
constexpr auto cfline = univariate::idx_msm_lines<F, std::vector<F>>;
constexpr auto cfcol = univariate::idx_msm_cols<F, std::vector<F>>;
constexpr auto cfdiag = univariate::idx_msm_diag<F, std::vector<F>>;
constexpr F PINF = utils::PINF<F>;
constexpr F QNAN = utils::QNAN<F>;

F msm(std::vector<F> const& lines, std::vector<F> const& cols, F cost, F cutoff) {
  return msm(
    lines.size(), cols.size(), cfline(lines, cols, cost), cfcol(lines, cols, cost), cfdiag(lines, cols), cutoff
  );
}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace ref {

  /// Implementing cost from the original paper
  /// c is the minimal cost of an operation
  inline double get_cost(double xi, double xi1, double yi, double c) {
    if ((xi1<=xi&&xi<=yi)||(xi1>=xi&&xi>=yi)) {
      return c;
    } else {
      return c + std::min(std::fabs(xi - xi1), std::fabs(xi - yi));
    }
  }

  /// Naive MSM with a window. Reference code.
  double msm_matrix(const std::vector<double>& series1, const std::vector<double>& series2, double c) {
    const long length1 = utils::to_signed(series1.size());
    const long length2 = utils::to_signed(series2.size());

    // Check lengths. Be explicit in the conditions.
    if (length1==0&&length2==0) { return 0; }
    if (length1==0&&length2!=0) { return PINF; }
    if (length1!=0&&length2==0) { return PINF; }

    const long maxLength = std::max(length1, length2);
    std::vector<std::vector<double>> cost(maxLength, std::vector<double>(maxLength, PINF));

    // Initialization
    cost[0][0] = std::abs(series1[0] - series2[0]);
    for (long i = 1; i<length1; i++) {
      cost[i][0] = cost[i - 1][0] + get_cost(series1[i], series1[i - 1], series2[0], c);
    }
    for (long i = 1; i<length2; i++) {
      cost[0][i] = cost[0][i - 1] + get_cost(series2[i], series1[0], series2[i - 1], c);
    }

    // Main Loop
    for (long i = 1; i<length1; i++) {
      for (long j = 1; j<length2; j++) {
        double d1, d2, d3;
        d1 = cost[i - 1][j - 1] + std::abs(series1[i] - series2[j]);                    // Diag
        d2 = cost[i - 1][j] + get_cost(series1[i], series1[i - 1], series2[j], c);      // Prev
        d3 = cost[i][j - 1] + get_cost(series2[j], series1[i], series2[j - 1], c);      // Top
        cost[i][j] = utils::min(d1, d2, d3);
      }
    }

    // Output
    return cost[length1 - 1][length2 - 1];
  }

}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Univariate MSM Fixed length", "[msm][univariate]") {
  // Setup univariate with fixed length
  mock::Mocker mocker;
  const auto& msm_costs = mocker.msm_costs;
  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("MSM(s,s) == 0") {
    for (const auto& s : fset) {
      for (auto c : msm_costs) {
        const double msm_ref_v = ref::msm_matrix(s, s, c);
        REQUIRE(msm_ref_v==0);

        const auto msm_v = msm(s, s, c, PINF);
        REQUIRE(msm_v==0);
      }
    }
  }

  SECTION("MSM(s1, s2)") {
    for (size_t i = 0; i<nbitems - 1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i + 1];

      for (auto c : msm_costs) {
        const double msm_ref_v = ref::msm_matrix(s1, s2, c);
        INFO("Exact same operation order. Expect exact floating point equality.");

        const auto msm_tempo = msm(s1, s2, c, PINF);
        REQUIRE(msm_ref_v==msm_tempo);
      }
    }
  }

  SECTION("NN1 MSM") {
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
        // Create the univariate squared Euclidean distance for our msm functions
        for (auto c : msm_costs) {
          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = ref::msm_matrix(s1, s2, c);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }
          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = msm(s1, s2, c, PINF);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }
          REQUIRE(idx_ref==idx);
          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_tempo = msm(s1, s2, c, bsf_tempo);
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

TEST_CASE("Univariate MSM Variable length", "[msm][univariate]") {
  // Setup univariate dataset with varying length
  mock::Mocker mocker;
  const auto& msm_costs = mocker.msm_costs;
  const auto fset = mocker.vec_rs_randvec(nbitems);

  SECTION("MSM(s,s) == 0") {
    for (const auto& s : fset) {
      for (auto c : msm_costs) {
        const double msm_ref_v = ref::msm_matrix(s, s, c);
        REQUIRE(msm_ref_v==0);

        const auto msm_v = msm(s, s, c, PINF);
        REQUIRE(msm_v==0);
      }
    }
  }

  SECTION("MSM(s1, s2)") {
    for (size_t i = 0; i<nbitems - 1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i + 1];
      for (auto c : msm_costs) {
        const double msm_ref_v = ref::msm_matrix(s1, s2, c);
        INFO("Exact same operation order. Expect exact floating point equality.");

        const auto msm_tempo_v = msm(s1, s2, c, QNAN);
        REQUIRE(msm_ref_v==msm_tempo_v);
      }
    }
  }

  SECTION("NN1 MSM") {
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
        // Create the univariate squared Euclidean distance for our msm functions
        for (auto c : msm_costs) {
          // --- --- --- --- --- --- --- --- --- --- --- ---
          const double v_ref = ref::msm_matrix(s1, s2, c);
          if (v_ref<bsf_ref) {
            idx_ref = j;
            bsf_ref = v_ref;
          }
          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v = msm(s1, s2, c, PINF);
          if (v<bsf) {
            idx = j;
            bsf = v;
          }
          REQUIRE(idx_ref==idx);
          // --- --- --- --- --- --- --- --- --- --- --- ---
          const auto v_tempo = msm(s1, s2, c, bsf_tempo);
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
