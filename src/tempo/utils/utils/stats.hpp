#pragma once

#include <cstddef>
#include <cmath>

namespace tempo::utils {

  /** Compute a running mean, variance and standard deviation following Welford method
   * (which avoids some floating point errors).
   * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm
   **/
  class StddevWelford {
    std::size_t n{0};
    double online_mean{0};
    double online_variance{0};
  public:

    inline void update(double x) {
      ++n;
      double delta = x - online_mean;
      online_mean += delta/(double)n;
      double delta2 = x - online_mean;
      online_variance += delta*delta2;
    }

    [[nodiscard]] inline double get_mean() const { return online_mean; }

    /** Get the variance for a population.
     * Valid if at least 2 values have been added. */
    [[nodiscard]] inline double get_variance_p() const {
      if (n<2) { return std::nan(""); }
      else { return online_variance/(double)n; }
    }

    /** Get the variance for a sample (subset of a population), applying Bessel's correction.
     * Valid if at least 2 values have been added. */
    [[nodiscard]] inline double get_variance_s() const {
      if (n<2) { return std::nan(""); }
      else { return online_variance/double(n - 1); }
    }

    /** Get the standard deviation for a population.
     * Valid if at least 2 values have been added */
    [[nodiscard]] inline double get_stddev_p() const { return std::sqrt(get_variance_p()); }

    /** Get the standard deviation for a sample (subset of a population), applying Bessel's correction.
     * Valid if at least 2 values have been added */
    [[nodiscard]] inline double get_stddev_s() const { return std::sqrt(get_variance_s()); }
  };

}