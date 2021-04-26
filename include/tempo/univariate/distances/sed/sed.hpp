#pragma once

#include "../../../tseries/tseries.hpp"
#include "../../../utils/utils.hpp"
#include "../distances.hpp"

namespace tempo::univariate {

  namespace internal {

    template<typename FloatType, auto dist = square_dist<FloatType>>
    [[nodiscard]] inline FloatType sed(
        const FloatType *lines, size_t nblines,
        const FloatType *cols, size_t nbcols,
        const FloatType *weights,
        const double diagweight_,
        const FloatType cutoff
    ) {
      constexpr double penalty = 1.0;
      const double diagweight = diagweight_;
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // In debug mode, check preconditions
      assert(lines != nullptr && nblines != 0 && nblines < MAX_SERIES_LENGTH);
      assert(cols != nullptr && nbcols != 0 && nbcols < MAX_SERIES_LENGTH);
      assert(nbcols <= nblines);
      // Adapt constants to the floating point type
      constexpr auto POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<FloatType>;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      const FloatType ub = nextafter(cutoff, POSITIVE_INFINITY) - dist(lines[nblines - 1], cols[nbcols - 1]);

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Double buffer allocation, no initialisation required (border condition manage in the code).
      // Base indices for the 'c'urrent row and the 'p'revious row.
      auto buffers = std::unique_ptr<FloatType[]>(new FloatType[nbcols * 2]);
      size_t c{0}, p{nbcols};

      // Line & column counters
      size_t i{0}, j{0};

      // Cost accumulator. Also used as the "left neighbour".
      FloatType cost;

      // EAP variables: track where to start the next line, and the position of the previous pruning point.
      // Must be init to 0: index 0 is the next starting index and also the "previous pruning point"
      size_t next_start{0}, prev_pp{0};

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Initialisation of the first line.
      {
        const FloatType l0 = lines[0];
        // Fist cell is a special case.
        // Check against the original upper bound dealing with the case where we have both series of length 1.
        cost = dist(l0, cols[0]);
        if (cost > cutoff) { return POSITIVE_INFINITY; }
        buffers[c + 0] = cost;
        // All other cells. Checking against "ub" is OK as the only case where the last cell of this line is the
        // last alignment is taken are just above (1==nblines==nbcols, and we have nblines >= nbcols).
        size_t curr_pp = 1;
        for (j = 1; j == curr_pp && j < nbcols; ++j) {
          cost = cost + dist(l0, cols[j]) + penalty*weights[absdiff(i,j)]; // Left: penalty
          buffers[c + j] = cost;
          if (cost <= ub) { ++curr_pp; }
        }
        ++i;
        prev_pp = curr_pp;
      }

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Main loop
      for (; i < nblines; ++i) {
        // --- --- --- Swap and variables init
        std::swap(c, p);
        const FloatType li = lines[i];
        size_t curr_pp = next_start; // Next pruning point init at the start of the line
        j = next_start;
        // --- --- --- Stage 0: Special case for the first column. Can only look up (border on the left)
        {
          cost = buffers[p + j] + dist(li, cols[j]) + penalty*weights[absdiff(i,j)]; // Top: penalty
          buffers[c + j] = cost;
          if (cost <= ub) { curr_pp = j + 1; } else { ++next_start; }
          ++j;
        }
        // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        for (; j == next_start && j < prev_pp; ++j) {
          const auto d = dist(li, cols[j]);
              cost = std::min(
                  d*diagweight + buffers[p + j - 1],               // Diag: no penalty
                  d+buffers[p + j] + penalty*weights[absdiff(i,j)]  // Top: penalty
                  );
          buffers[c + j] = cost;
          if (cost <= ub) { curr_pp = j + 1; } else { ++next_start; }
        }
        // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        for (; j < prev_pp; ++j) {
          const auto d = dist(li, cols[j]);
          cost = min(d + cost + penalty*weights[absdiff(i,j)],              // Left: penalty
                     d*diagweight +                   buffers[p + j - 1],          // Diag: no penalty
                     d +                    buffers[p + j] + penalty*weights[absdiff(i,j)]);   // Top: penalty
          buffers[c + j] = cost;
          if (cost <= ub) { curr_pp = j + 1; }
        }
        // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if (j < nbcols) { // If so, two cases.
          const auto d = dist(li, cols[j]);
          if (j == next_start) { // Case 1: Advancing next start: only diag (no penalty)
            cost = buffers[p + j - 1] + d*diagweight;
            buffers[c + j] = cost;
            if (cost <= ub) { curr_pp = j + 1; }
            else {
              // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
              if (i == nblines - 1 && j == nbcols - 1 && cost <= cutoff) { return cost; }
              else { return POSITIVE_INFINITY; }
            }
          } else { // Case 2: Not advancing next start: possible path in previous cells: left (penalty) and diag.
            cost = std::min(d+cost + penalty*weights[absdiff(i,j)],
                            d*diagweight + buffers[p + j - 1]);
            buffers[c + j] = cost;
            if (cost <= ub) { curr_pp = j + 1; }
          }
          ++j;
        } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
          if (j == next_start) {
            // But only if we are above the original UB
            // Else set the next starting point to the last valid column
            if (cost > cutoff) { return POSITIVE_INFINITY; }
            else { next_start = nbcols - 1; }
          }
        }
        // --- --- --- Stage 4: After the previous pruning point: only prev.
        // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
        for (; j == curr_pp && j < nbcols; ++j) {
          const auto d = dist(li, cols[j]);
          cost = cost + d + penalty*weights[absdiff(i,j)]; // Left: penalty
          buffers[c + j] = cost;
          if (cost <= ub) { ++curr_pp; }
        }
        // --- --- ---
        prev_pp = curr_pp;
      } // End of main loop for(;i<nblines;++i)

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Finalisation
      // Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
      if (j == nbcols && cost <= cutoff) { return cost; }
      else { return POSITIVE_INFINITY; }
    }

  } // End of namespace internal

  template<typename FloatType, auto dist = square_dist<FloatType>>
  [[nodiscard]] FloatType sed(
      const FloatType *series1, size_t length1,
      const FloatType *series2, size_t length2,
      const FloatType *weights,
      const double diagweight
  ) {
    const auto check_result = check_order_series(series1, length1, series2, length2);
    switch (check_result.index()) {
      case 0: {
        return std::get<0>(check_result);
      }
      case 1: {
        const auto[lines, nblines, cols, nbcols] = std::get<1>(check_result);

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Compute a cutoff point using the diagonal
        FloatType cutoff{0};
        // We have less columns than lines: cover all the columns first.
        for (size_t i{0}; i < nbcols; ++i) { cutoff += dist(lines[i], cols[i]); }
        // Then go down in the last column
        if (nbcols < nblines) {
          const auto lc = cols[nbcols - 1];
          for (size_t i{nbcols}; i < nblines; ++i) { cutoff += dist(lines[i], lc); }
        }

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        return internal::sed<FloatType, dist>(lines, nblines, cols, nbcols, weights, diagweight, cutoff);
      }
      default:
        should_not_happen();
    }
  }

  /// Helper for the above, using vectors
  template<typename FloatType, auto dist = square_dist<FloatType>>
  [[nodiscard]] inline FloatType sed(const std::vector<FloatType> &series1, const std::vector<FloatType> &series2,
                                     const FloatType *weights,
                                     const double diagweight
                                     ) {
    return sed<FloatType, dist>(series1.data(), series1.size(), series2.data(), series2.size(), diagweight, weights);
  }

  /// Helper for the above, using TSeries
  template<typename FloatType, typename LabelType, auto dist = square_dist<FloatType>>
  [[nodiscard]] inline FloatType sed(
      const TSeries<FloatType, LabelType> &series1,
      const TSeries<FloatType, LabelType> &series2,
      const FloatType *weights,
      const double diagweight
  ) {
    return sed<FloatType, dist>(series1.data(), series1.length(), series2.data(), series2.length(), diagweight, weights);
  }

  /// Build a distfun_t for the above
  template<typename FloatType, typename LabelType, auto dist = square_dist<FloatType>>
  [[nodiscard]] inline distfun_t<FloatType, LabelType> distfun_sed(const FloatType *weights, const double diagweight) {
    return distfun_t<FloatType, LabelType>{
        [weights, diagweight](
            const TSeries<FloatType, LabelType> &series1,
            const TSeries<FloatType, LabelType> &series2
        ) { return sed<FloatType, LabelType, dist>(series1, series2, weights, diagweight); }
    };
  }

  template<typename FloatType, auto dist = square_dist<FloatType>>
  [[nodiscard]] FloatType sed(
      const FloatType *series1, size_t length1,
      const FloatType *series2, size_t length2,
      const FloatType *weights,
      const double diagweight,
      FloatType cutoff
  ) {
    const auto check_result = check_order_series(series1, length1, series2, length2);
    switch (check_result.index()) {
      case 0: {
        return std::get<0>(check_result);
      }
      case 1: {
        const auto[lines, nblines, cols, nbcols] = std::get<1>(check_result);
        return internal::sed<FloatType, dist>(lines, nblines, cols, nbcols, weights, diagweight, cutoff);
      }
      default:
        should_not_happen();
    }
  }

  /// Helper for the above, using vectors
  template<typename FloatType, auto dist = square_dist<FloatType>>
  [[nodiscard]] inline FloatType sed(
      const std::vector<FloatType> &series1,
      const std::vector<FloatType> &series2,
      const FloatType *weights,
      const double diagweight,
      FloatType cutoff) {
    return sed<FloatType, dist>(series1.data(), series1.size(), series2.data(), series2.size(), weights, diagweight, cutoff);
  }

  /// Helper for the above, using TSeries
  template<typename FloatType, typename LabelType, auto dist = square_dist<FloatType>>
  [[nodiscard]] inline FloatType sed(
      const TSeries<FloatType, LabelType> &series1,
      const TSeries<FloatType, LabelType> &series2,
      const FloatType *weights,
      const FloatType diagweight,
      FloatType cutoff) {
    return sed<FloatType, dist>(series1.data(), series1.length(), series2.data(), series2.length(), weights, diagweight, cutoff);
  }

  /// Build a distfun_cutoff_t for the above
  template<typename FloatType, typename LabelType, auto dist = square_dist<FloatType>>
  [[nodiscard]] inline distfun_cutoff_t<FloatType, LabelType> distfun_cutoff_sed(
      std::shared_ptr<std::vector<FloatType>> weights,
      const double diagweight
      ) {
    return distfun_cutoff_t<FloatType, LabelType>{
        [weights, diagweight](
            const TSeries<FloatType, LabelType> &series1,
            const TSeries<FloatType, LabelType> &series2,
            FloatType co
        ) {
          return sed<FloatType, LabelType, dist>(series1, series2, weights->data(), diagweight, co);
        }
    };
  }

} // End of namespace tempo::univariate
