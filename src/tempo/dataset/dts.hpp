#pragma once

#include <tempo/utils/utils.hpp>
#include "tseries.hpp"
#include "dataset.hpp"

namespace tempo {

  /// Helper for Dataset of time series.
  using DTS = DataSplit<TSeries>;

  /// Map of named DTS
  using DTSMap = std::map<std::string, DTS>;

  /// Helper for a DTS (Dataset of Time Series), computing statistics per dimension
  struct DTS_Stats {
    arma::Col<F> _min;
    arma::Col<F> _max;
    arma::Col<F> _mean;
    arma::Col<F> _stddev;

    /// Compute statistic on a split subset
    DTS_Stats(const DTS& dts, const IndexSet& subset) {

      arma::running_stat_vec<arma::Col<F>> stat;
      for (const auto i : subset) {
        const TSeries& s = dts[i];
        const arma::Mat<F>& mat = s.matrix();
        for (size_t c = 0; c<mat.n_cols; ++c) {
          stat(mat.col(c));
        }
      }

      _min = stat.min();
      _max = stat.max();
      _mean = stat.mean();
      _stddev = stat.stddev(0); // norm_type=0 performs normalisation using N-1 (N=number of samples)
    }

    /// Compute statistics on a full split
    explicit DTS_Stats(const DTS& dts) : DTS_Stats(dts, IndexSet(dts.size())) {}

  };

  /// Helper for univariate DTS
  inline F stddev(const DTS& dts, const IndexSet& is) {
    DTS_Stats stat(dts, is);
    return stat._stddev[0];
  }

} // End of namespace tempo