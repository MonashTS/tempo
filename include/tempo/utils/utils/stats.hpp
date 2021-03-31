#pragma once

namespace tempo::stats {

  /** Given a collection of cardinalities, compute the Gini Impurity.
   * @tparam ForwardIterator Must be a forward iterator (used twice)
   * @param begin First item of the collection
   * @param end Marked the end of the collection
   * @return 0<=gi<1 where 0 means total purity (all item in one class).
   */
  template<typename ForwardIterator>
  double gini_impurity(ForwardIterator begin, ForwardIterator end) {
    // Ensure that we never encounter a "floating point near 0" issue.
    if (std::distance(begin, end)==1) { return 0; }
    // Gini impurity computation
    double total_size{0};
    for (auto it = begin; it!=end; ++it) { total_size += *it; }
    double sum{0};
    for (auto it = begin; it!=end; ++it) {
      double p = (*it)/total_size;
      sum += p*p;
    }
    return 1-sum;
  }



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
      double delta = x-online_mean;
      online_mean += delta/n;
      double delta2 = x-online_mean;
      online_variance += delta*delta2;
    }

    [[nodiscard]] inline double get_mean() { return online_mean; }

    /** Get the variance for a population.
     * Valid if at least 2 values have been added. */
    [[nodiscard]] inline double get_variance_p() {
      if (n<2) { return std::nan(""); }
      else { return online_variance/n; }
    }

    /** Get the variance for a sample.
     * Valid if at least 2 values have been added. */
    [[nodiscard]] inline double get_variance_s() {
      if (n<2) { return std::nan(""); }
      else { return online_variance/double(n-1); }
    }

    /** Get the standard deviation for a population.
     * Valid if at least 2 values have been added */
    [[nodiscard]] inline double get_stddev_p() { return std::sqrt(get_variance_p()); }

    /** Get the standard deviation for a sample.
     * Valid if at least 2 values have been added */
    [[nodiscard]] inline double get_stddev_s() { return std::sqrt(get_variance_s()); }
  };



} // End of namespace tempo