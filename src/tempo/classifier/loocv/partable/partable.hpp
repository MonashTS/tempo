#pragma once

#include "dist_interface.hpp"

#include <tempo/dataset/dts.hpp>

namespace tempo::classifier::nn1loocv {

  /// Nearest Neighbour Cell
  /// A cell of our table, with its own mutex
  struct NNC {
    std::mutex mutex{};   // Lock/unlock for multithreading
    size_t NNindex{};     // Index of the NN
    F NNdistance{};       // Distance to the NN
  };

  /// Search for the best parameter through LOOCV
  /// @param distance A distance computation function of type 'dist_fb'.
  ///     Must capture the series, the actual distance, and the parameters, so that it can be called with indexes.
  /// @param distanceUB Simular as above, of type 'distUB_fb', producing an upper bound.
  /// @param nbtrain Number of train exemplars: the distances will be call with distance(i, j, p) with
  ///     0<=i<nbtrain, 0<=j<nbtrain, i!=j, and p a parameter index
  /// @param nbparams Number of parameters
  ///     It is assumed that the distance functions capture a mapping of parameter N->Internal Parameter,
  ///     usually with a vector of parameter p, with 0 <= n:N < nbparams.
  ///     It is assumed that the parameter are ordered such that the distance computed with p[0] is a LB for p[1], etc:
  ///         dist(a, b, 0) <= dist(a, b, 1) ... <= dist(a, b, n-1)
  ///     In turns, p[last] produces an upper bound for p[last-1] ... etc ... for p[0]
  ///
  ///     Example with DTW: dtw(a, b, FULL WINDOW) <= dist(a, b, WINDOW = 0)
  ///     Hence, the windows must be ordered in decreasing parameter order, i.e. p[0]  >=  p[1]
  ///
  ///     Example with ADTW: adtw(a, b, SMALL PENALTY) <= dist(a, b, LARGE PENALTY)
  ///     Hence, the penalties must be ordered in increasing parameter order, i.e. p[0]  =<  p[1]
  /// @param nbthreads parallelize the process on nbthreads -
  ///     Note that if nbthreads<2, this is not the best method as we spend time taking/realising mutexes!
  /// @return (vector of best parameters' index, bestError)
  std::tuple<std::vector<size_t>, size_t> partable(
    distParam_ft distance,
    distUB_ft distanceUB,
    DatasetHeader const& train_header,
    size_t nbparams,
    size_t nbthreads,
    std::ostream *out = nullptr
  );

  /// Given a i_LOOCVDist, search for the best possible parameterization and test it.
  /// Note: set the result on the incoming i_LOOCVDist instance
  void partable(
    i_LOOCVDist& instance,
    size_t nbtrain, // May be smaller than train_header.size() for a subset
    DatasetHeader const& train_header,
    size_t nbtest,  // May be smaller than test_header.size() for a subset
    DatasetHeader const& test_header,
    PRNG& prng,
    size_t nbthreads,
    std::ostream *out = nullptr
  );

}// End of namespace tempo::classifier::nn1loocv
