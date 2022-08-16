#include <catch2/catch_test_macros.hpp>

#include "erp.hpp"

#include <mock/mockseries.hpp>
#include <vector>

using namespace tempo::distance;

using F = double;

constexpr size_t nbitems = 500;
constexpr auto cfun = univariate::idx_ad2<F, std::vector<F>>;
constexpr auto gvcfun = univariate::idx_gvad2<F, std::vector<F>>;
constexpr F PINF = utils::PINF<F>;
constexpr F QNAN = std::numeric_limits<F>::quiet_NaN();

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Reference
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
namespace ref {

  /// Cost function squared euclidean distance
  double sqcfun(double a, double b) {
    double d = a - b;
    return d*d;
  }

  /// Naive ERP with a window. Reference code.
  double erp_matrix(const std::vector<double>& series1, const std::vector<double>& series2, size_t w, double gValue) {
    const long length1 = utils::to_signed(series1.size());
    const long length2 = utils::to_signed(series2.size());
    long ww = utils::to_signed(w);

    // Check lengths. Be explicit in the conditions.
    if (length1==0&&length2==0) { return 0; }
    if (length1==0&&length2!=0) { return PINF; }
    if (length1!=0&&length2==0) { return PINF; }

    // We will only allocate a double-row buffer: use the smallest possible dimension as the columns.
    const std::vector<double>& cols = (length1<length2) ? series1 : series2;
    const std::vector<double>& lines = (length1<length2) ? series2 : series1;
    const long nbcols = std::min(length1, length2);
    const long nblines = std::max(length1, length2);

    // Cap the windows
    if (ww>nblines) { ww = nblines; }

    // Check if, given the constraint w, we can have an alignment.
    if (nblines - nbcols>ww) { return PINF; }

    // Allocate a double buffer for the columns. Declare the index of the 'c'urrent and 'p'revious buffer.
    // Note: we use a vector as a way to initialize the buffer with PINF
    std::vector<std::vector<double>> matrix(nblines + 1, std::vector<double>(nbcols + 1, PINF));

    // Initialisation of the first line and column
    matrix[0][0] = 0;
    for (long j{1}; j<nbcols + 1; j++) {
      matrix[0][j] = matrix[0][j - 1] + sqcfun(gValue, cols[j - 1]);
    }
    for (long i{1}; i<nblines + 1; i++) {
      matrix[i][0] = matrix[i - 1][0] + sqcfun(lines[i - 1], gValue);
    }

    // Iterate over the lines
    for (long i{1}; i<nblines + 1; ++i) {
      const double li = lines[i - 1];
      long l = std::max<long>(i - ww, 1);
      long r = std::min<long>(i + ww + 1, nbcols + 1);

      // Iterate through the rest of the columns
      for (long j{l}; j<r; ++j) {
        matrix[i][j] = utils::min(
          matrix[i][j - 1] + sqcfun(gValue, cols[j - 1]),    // Previous
          matrix[i - 1][j - 1] + sqcfun(li, cols[j - 1]),    // Diagonal
          matrix[i - 1][j] + sqcfun(li, gValue)              // Above
        );
      }
    } // End of for over lines

    return matrix[nblines][nbcols];
  }

}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
TEST_CASE("Univariate ERP Fixed length", "[erp][univariate]") {
  // Setup univariate with fixed length
  mock::Mocker mocker;
  const auto& wratios = mocker.wratios;
  const auto& gvalues = mocker.gvalues;

  const auto fset = mocker.vec_randvec(nbitems);

  SECTION("ERP(s,s) == 0") {
    for (const auto& s : fset) {
      for (double wr : wratios) {
        auto w = (size_t)(wr*(double)mocker._fixl);
        for (auto gv : gvalues) {
          const double erp_ref_v = ref::erp_matrix(s, s, w, gv);
          REQUIRE(erp_ref_v==0);

          const auto erp_v = erp(s.size(), s.size(), gvcfun(s, gv), gvcfun(s, gv), cfun(s, s), w, PINF);
          REQUIRE(erp_v==0);
        }
      }
    }
  }

  SECTION("ERP(s1, s2)") {
    for (size_t i = 0; i<nbitems - 1; ++i) {
      const auto& s1 = fset[i];
      const auto& s2 = fset[i + 1];

      for (double wr : wratios) {
        const auto w = (size_t)(wr*(double)mocker._fixl);

        for (auto gv : gvalues) {

          const double erp_ref_v = ref::erp_matrix(s1, s2, w, gv);
          INFO("Exact same operation order. Expect exact floating point equality.");

          const auto erp_tempo = erp(s1.size(), s2.size(), gvcfun(s1, gv), gvcfun(s2, gv), cfun(s1, s2), w, PINF);
          REQUIRE(erp_ref_v==erp_tempo);
        }
      }
    }
  }

  SECTION("NN1 ERP") {
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
      size_t idx_eap = 0;
      double bsf_eap = PINF;

      // NN1 loop
      for (size_t j = 0; j<nbitems; j += 5) {
        // Skip self.
        if (i==j) { continue; }
        const auto& s2 = fset[j];
        // Create the univariate squared Euclidean distance for our erp functions
        for (double wr : wratios) {
          const auto w = (size_t)(wr*(double)mocker._fixl);

          for (auto gv : gvalues) {
            // --- --- --- --- --- --- --- --- --- --- --- ---
            const double v_ref = ref::erp_matrix(s1, s2, w, gv);
            if (v_ref<bsf_ref) {
              idx_ref = j;
              bsf_ref = v_ref;
            }

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v = erp(s1.size(), s2.size(), gvcfun(s1, gv), gvcfun(s2, gv), cfun(s1, s2), w, PINF);
            if (v<bsf) {
              idx = j;
              bsf = v;
            }

            REQUIRE(idx_ref==idx);

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v_eap = erp(s1.size(), s2.size(), gvcfun(s1, gv), gvcfun(s2, gv), cfun(s1, s2), w, bsf_eap);
            if (v_eap<bsf_eap) {
              idx_eap = j;
              bsf_eap = v_eap;
            }

            REQUIRE(idx_ref==idx_eap);
          }
        }
      }
    }// End query loop
  }// End section

}

TEST_CASE("Univariate ERP Variable length", "[erp][univariate]") {
  // Setup univariate dataset with varying length
  mock::Mocker mocker;
  const auto& wratios = mocker.wratios;
  const auto& gvalues = mocker.gvalues;

  const auto fset = mocker.vec_rs_randvec(nbitems);

  SECTION("ERP(s,s) == 0") {
    for (const auto& s : fset) {
      for (double wr : wratios) {
        const auto w = (size_t)(wr*(double)s.size());
        for (auto gv : gvalues) {
          const double erp_ref_v = ref::erp_matrix(s, s, w, gv);
          REQUIRE(erp_ref_v==0);

          const auto erp_v = erp(s.size(), s.size(), gvcfun(s, gv), gvcfun(s, gv), cfun(s, s), w, PINF);
          REQUIRE(erp_v==0);
        }
      }
    }
  }

  SECTION("ERP(s1, s2)") {
    for (size_t i = 0; i<nbitems - 1; ++i) {
      for (double wr : wratios) {
        const auto& s1 = fset[i];
        const auto& s2 = fset[i + 1];
        const auto w = (size_t)(wr*(double)std::min<size_t>(s1.size(), s2.size()));
        for (auto gv : gvalues) {
          const double erp_ref_v = ref::erp_matrix(s1, s2, w, gv);
          INFO("Exact same operation order. Expect exact floating point equality.");

          const auto v_tempo = erp(s1.size(), s2.size(), gvcfun(s1, gv), gvcfun(s2, gv), cfun(s1, s2), w, QNAN);
          REQUIRE(erp_ref_v==v_tempo);
        }
      }
    }
  }

  SECTION("NN1 ERP") {
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

        for (double wr : wratios) {
          const auto w = (size_t)(wr*(double)std::min(s1.size(), s2.size()));

          for (auto gv : gvalues) {

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const double v_ref = ref::erp_matrix(s1, s2, w, gv);
            if (v_ref<bsf_ref) {
              idx_ref = j;
              bsf_ref = v_ref;
            }

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v = erp(s1.size(), s2.size(), gvcfun(s1, gv), gvcfun(s2, gv), cfun(s1, s2), w, PINF);
            if (v<bsf) {
              idx = j;
              bsf = v;
            }

            REQUIRE(idx_ref==idx);

            // --- --- --- --- --- --- --- --- --- --- --- ---
            const auto v_tempo = erp(s1.size(), s2.size(), gvcfun(s1, gv), gvcfun(s2, gv), cfun(s1, s2), w, bsf_tempo);
            if (v_tempo<bsf_tempo) {
              idx_tempo = j;
              bsf_tempo = v_tempo;
            }

            REQUIRE(idx_ref==idx_tempo);
          }
        }
      }
    }// End query loop
  }// End section
}
