#include <catch.hpp>
#include <tempo/univariate/distances/twe/twe.hpp>

#include "../tests_tools.hpp"
#include "references/twe/twe.hpp"

using namespace tempo::univariate;
constexpr double POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<double>;

/// Our own TWE reference code.
double twe_matrix(const std::vector<double>& series1, const std::vector<double>& series2, double nu, double lambda) {
  const size_t length1 = series1.size();
  const size_t length2 = series2.size();

  // Check lengths. Be explicit in the conditions.
  if (length1==0 && length2==0) { return 0; }
  if (length1==0 && length2!=0) { return POSITIVE_INFINITY; }
  if (length1!=0 && length2==0) { return POSITIVE_INFINITY; }

  const size_t maxLength = std::max(length1, length2);
  std::vector<std::vector<double>> matrix(maxLength, std::vector<double>(maxLength, POSITIVE_INFINITY));

  const double nu_lambda = nu+lambda;
  const double nu2 = 2*nu;

  // Initialization: first cell, first column and first row
  matrix[0][0] = square_dist(series1[0], series2[0]);
  for (size_t i = 1; i<length1; i++) { matrix[i][0] = matrix[i-1][0]+square_dist(series1[i], series1[i-1])+nu_lambda; }
  for (size_t j = 1; j<length2; j++) { matrix[0][j] = matrix[0][j-1]+square_dist(series2[j], series2[j-1])+nu_lambda; }

  // Main Loop
  for (size_t i = 1; i<length1; i++) {
    for (size_t j = 1; j<length2; j++) {
      // Top: over the lines
      double t = matrix[i-1][j]+square_dist(series1[i], series1[i-1])+nu_lambda;
      // Diagonal
      double d = matrix[i-1][j-1]+square_dist(series1[i], series2[j])+square_dist(series1[i-1], series2[j-1])+nu2*tempo::absdiff(i, j);
      // Previous: over the columns
      double p = matrix[i][j-1]+square_dist(series2[j], series2[j-1])+nu_lambda;
      //
      matrix[i][j] = std::min(t, std::min(d, p));
    }
  }

  // Output
  return matrix[length1-1][length2-1];
}


// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// Testing
// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

TEST_CASE("TWE Fixed length", "[twe]") {
  Catch::StringMaker<double>::precision = 17;

  // Create a random dataset
  constexpr int nbitems = ttools::def_nbitems;
  constexpr int fsize = ttools::def_fixed_size;
  const auto fset = ttools::get_set_fixed_length(ttools::prng, nbitems, fsize);


  SECTION("TWE debug") {
    const std::vector<double> italypd0 = {-0.71051757, -1.1833204, -1.3724416, -1.5930829, -1.4670021, -1.3724416,
                                          -1.0887599, 0.045966947, 0.92853223, 1.0861332, 1.2752543, 0.96005242,
                                          0.61333034, 0.014446758, -0.6474772, -0.26923494, -0.20619456, 0.61333034,
                                          1.3698149, 1.4643754, 1.054613, 0.58181015, 0.1720477, -0.26923494};
    const std::vector<double> italypd17 = {-0.25805705, -0.73143556, -1.1259177, -1.362607, -1.4809516, -1.4415034,
                                           -1.0864695, -0.77088377, -0.81033198, 0.018080433, 0.33366612, 0.64925181,
                                           0.68870002, -0.1002642, -0.21860884, -0.29750526, -0.25805705, 0.88594105,
                                           1.5171125, 1.7143535, 1.6354571, 1.3593196, 0.84649285, 0.29421791};
    const double nu0 = 0.00001;
    const double lambda0 = 0;
    double vref_marteau = reference::twe_Marteau(italypd0, italypd17, nu0, lambda0);
    double vref = twe_matrix(italypd0, italypd17, nu0, lambda0);
    double v = twe(italypd0, italypd17, nu0, lambda0);
    double veap = twe(italypd0, italypd17, nu0, lambda0, POSITIVE_INFINITY);
    REQUIRE(vref==v);
    REQUIRE(vref==veap);

  }

  SECTION("TWE(s,s) == 0") {
    for (auto nu: ttools::twe_nus) {
      for (auto lambda: ttools::twe_lambdas) {
        for (const auto& series: fset) {
          double vref = reference::twe_Marteau(series, series, nu, lambda);
          REQUIRE(vref==0);

          double v = twe(series, series, nu, lambda);
          REQUIRE(v==0);

          double v_eap = twe(series, series, nu, lambda, POSITIVE_INFINITY);
          REQUIRE(v_eap==0);
        }
      }
    }
  }

  SECTION("TWE(s1, s2)") {
    for (auto nu: ttools::twe_nus) {
      for (auto lambda: ttools::twe_lambdas) {
        for (int i = 0; i<nbitems; i += 2) {
          const auto& series1 = fset[i];
          const auto& series2 = fset[i+1];

          double vref = reference::twe_Marteau(series1, series2, nu, lambda);
          INFO("Not exact same operation orders. Requires approximative equality.")

          double v = twe(series1, series2, nu, lambda);
          REQUIRE(vref==Approx(v));

          double v_eap = twe(series1, series2, nu, lambda, POSITIVE_INFINITY);
          REQUIRE(v==v_eap);
        }
      }
    }
  }

  SECTION("NN1 TWE") {

    for (auto nu: ttools::twe_nus) {
      for (auto lambda: ttools::twe_lambdas) {
        // Query loop
        for (int i = 0; i<nbitems; i += 3) {
          // Ref Variables
          int idx_ref = 0;
          double bsf_ref = POSITIVE_INFINITY;

          // Base Variables
          int idx = 0;
          double bsf = POSITIVE_INFINITY;

          // EAP Variables
          int idx_eap = 0;
          double bsf_eap = POSITIVE_INFINITY;

          // NN1 loop
          for (int j = 0; j<nbitems; j += 5) {
            // Skip self.
            if (i==j) { continue; }

            // --- --- --- --- --- --- --- --- --- --- --- ---
            double v_ref = reference::twe_Marteau(fset[i], fset[j], nu, lambda);
            if (v_ref<bsf_ref) {
              idx_ref = j;
              bsf_ref = v_ref;
            }

            // --- --- --- --- --- --- --- --- --- --- --- ---
            double v = twe(fset[i], fset[j], nu, lambda);
            if (v<bsf) {
              idx = j;
              bsf = v;
            }

            REQUIRE(idx_ref==idx);

            // --- --- --- --- --- --- --- --- --- --- --- ---
            double v_eap = twe(fset[i], fset[j], nu, lambda, bsf_eap);
            if (v_eap<bsf_eap) {
              idx_eap = j;
              bsf_eap = v_eap;
            }

            REQUIRE(idx_ref==idx_eap);
          }
        }// End query loop
      }
    }
  }// End section
}

TEST_CASE("TWE variable length", "[twe]") {

  // Create a random dataset
  constexpr int nbitems = ttools::def_nbitems;
  const auto fset = ttools::get_set_variable_length(ttools::prng, nbitems);

  SECTION("TWE(s,s) == 0") {
    for (auto nu: ttools::twe_nus) {
      for (auto lambda: ttools::twe_lambdas) {
        for (const auto& series: fset) {
          double vref = reference::twe_Marteau(series, series, nu, lambda);
          REQUIRE(vref==0);

          double v = twe(series, series, nu, lambda);
          REQUIRE(v==0);

          double v_eap = twe(series, series, nu, lambda, POSITIVE_INFINITY);
          REQUIRE(v_eap==0);
        }
      }
    }
  }

  SECTION("TWE(s1, s2)") {
    for (auto nu: ttools::twe_nus) {
      for (auto lambda: ttools::twe_lambdas) {
        for (int i = 0; i<nbitems; i += 2) {
          const auto& series1 = fset[i];
          const auto& series2 = fset[i+1];

          double vref = reference::twe_Marteau(series1, series2, nu, lambda);
          INFO("Not exact same operation orders. Requires approximated equality.")

          double v = twe(series1, series2, nu, lambda);
          REQUIRE(vref==Approx(v));

          double v_eap = twe(series1, series2, nu, lambda, POSITIVE_INFINITY);
          REQUIRE(v==v_eap);
        }
      }
    }
  }

  SECTION("NN1 TWE") {

    for (auto nu: ttools::twe_nus) {
      for (auto lambda: ttools::twe_lambdas) {
        // Query loop
        for (int i = 0; i<nbitems; i += 3) {

          // Ref Variables
          int idx_ref = 0;
          double bsf_ref = POSITIVE_INFINITY;

          // Base Variables
          int idx = 0;
          double bsf = POSITIVE_INFINITY;

          // EAP Variables
          int idx_eap = 0;
          double bsf_eap = POSITIVE_INFINITY;

          // NN1 loop
          for (int j = 0; j<nbitems; j += 5) {
            // Skip self.
            if (i==j) { continue; }

            // --- --- --- --- --- --- --- --- --- --- --- ---
            double v_ref = reference::twe_Marteau(fset[i], fset[j], nu, lambda);
            if (v_ref<bsf_ref) {
              idx_ref = j;
              bsf_ref = v_ref;
            }

            // --- --- --- --- --- --- --- --- --- --- --- ---
            double v = twe(fset[i], fset[j], nu, lambda);
            if (v<bsf) {
              idx = j;
              bsf = v;
            }

            REQUIRE(idx_ref==idx);

            // --- --- --- --- --- --- --- --- --- --- --- ---
            double v_eap = twe(fset[i], fset[j], nu, lambda, bsf_eap);
            if (v_eap<bsf_eap) {
              idx_eap = j;
              bsf_eap = v_eap;
            }

            REQUIRE(idx_ref==idx_eap);
          }
        }// End query loop
      }
    }
  }// End section
}
