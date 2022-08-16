#pragma once

#include "../utils.private.hpp"

namespace tempo::distance {

  namespace internal {

    /** Amerced Dynamic Time Warping (ADTW), Early Abandoned and Pruned (EAP).
     * @tparam F            Floating type used for the computation
     * @param nblines       Length of the line series.
     * @param nbcols        Length of the column series.
     * @param cfun          Indexed Cost function between two points
     * @param penalty       Fixed cost penalty for warping steps; must be >=0
     * @param cutoff        EAP cutoff; Attempt to prune computation of alignments with cost > cutoff.
     *                      May lead to early abandoning.
     * @param buffers_v     Buffer used to perform the computation. Will reallocate if required.
     * @return ADTW between the two series or +PINF if early abandoned.
     */
    template<typename F>
    F adtw(const size_t nblines,
           const size_t nbcols,
           utils::ICFun<F> auto cfun,
           const F penalty,
           const F cutoff,
           std::vector<F>& buffer_v) {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // in debug mode, check preconditions
      assert(nblines!=0);
      assert(nbcols!=0);

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Adapt constants to the floating point type
      using utils::min;
      constexpr F PINF = utils::PINF<F>;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds.
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      const F ub = nextafter(cutoff, PINF) - cfun(nblines - 1, nbcols - 1);

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Double buffer allocation, no initialisation required (border condition manage in the code).
      // Base indices for the 'c'urrent row and the 'p'revious row.
      buffer_v.assign(nbcols*2, 0);
      auto *buffer = buffer_v.data();
      size_t c{0}, p{nbcols};

      // Line & column counters
      size_t i{0}, j{0};

      // Cost accumulator. Also used as the "left neighbour".
      F cost;

      // EAP variables: track where to start the next line, and the position of the previous pruning point.
      // Must be init to 0: index 0 is the next starting index and also the "previous pruning point"
      size_t next_start{0}, prev_pp{0};

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Initialisation of the first line.
      {
        // Fist cell is a special case.
        // Check against the original upper bound dealing with the case where we have both series of length 1.
        cost = cfun(0, 0);
        if (cost>cutoff) { return PINF; }
        buffer[c + 0] = cost;
        // All other cells. Checking against "ub" is OK as the only case where the last cell of this line is the
        // last alignment is taken are just above (1==nblines==nbcols, and we have nblines >= nbcols).
        size_t curr_pp = 1;
        for (j = 1; j==curr_pp&&j<nbcols; ++j) {
          cost = cost + cfun(0, j) + penalty; // Left: penalty
          buffer[c + j] = cost;
          if (cost<=ub) { ++curr_pp; }
        }
        ++i;
        prev_pp = curr_pp;
      }

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Main loop
      for (; i<nblines; ++i) {
        // --- --- --- Swap and variables init
        std::swap(c, p);
        size_t curr_pp = next_start; // Next pruning point init at the start of the line
        j = next_start;
        // --- --- --- Stage 0: Special case for the first column. Can only look up (border on the left)
        {
          cost = buffer[p + j] + cfun(i, j) + penalty; // Top: penalty
          buffer[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
          ++j;
        }
        // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        for (; j==next_start&&j<prev_pp; ++j) {
          const auto d = cfun(i, j);
          cost = std::min(
            buffer[p + j - 1] + d,         // Diag: no penalty
            buffer[p + j] + d + penalty     // Top: penalty
          );
          buffer[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
        }
        // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        for (; j<prev_pp; ++j) {
          const auto d = cfun(i, j);
          cost = min(d + cost + penalty,   // Left: penalty
            buffer[p + j - 1] + d,         // Diag: no penalty
            buffer[p + j] + d + penalty
          );   // Top: penalty
          buffer[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; }
        }
        // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if (j<nbcols) { // If so, two cases.
          const auto d = cfun(i, j);
          if (j==next_start) { // Case 1: Advancing next start: only diag (no penalty)
            cost = buffer[p + j - 1] + d;
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
            else {
              // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
              if (i==nblines - 1&&j==nbcols - 1&&cost<=cutoff) { return cost; }
              else {
                return PINF;
              }
            }
          } else { // Case 2: Not advancing next start: possible path in previous cells: left (penalty) and diag.
            cost = std::min(cost + d + penalty, buffer[p + j - 1] + d);
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
          }
          ++j;
        } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
          if (j==next_start) {
            // But only if we are above the original UB
            // Else set the next starting point to the last valid column
            if (cost>cutoff) {
              return PINF;
            } else { next_start = nbcols - 1; }
          }
        }
        // --- --- --- Stage 4: After the previous pruning point: only prev.
        // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
        for (; j==curr_pp&&j<nbcols; ++j) {
          const auto d = cfun(i, j);
          cost = cost + d + penalty; // Left: penalty
          buffer[c + j] = cost;
          if (cost<=ub) { ++curr_pp; }
        }
        // --- --- ---
        prev_pp = curr_pp;
      } // End of main loop for(;i<nblines;++i)

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Finalisation
      // Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
      if (j==nbcols&&cost<=cutoff) { return cost; } else { return PINF; }
    }

  } // End of namespace internal

  /** Amerced Dynamic Time Warping (ADTW) Early Abandoned and Pruned (EAP).
   * @tparam F            Floating type used for the computation
   * @param length1       Length of the first series.
   * @param length2       Length of the second series.
   * @param cfun          Indexed Cost function between two points
   * @param penalty       Fixed cost penalty for warping steps; must be >=0
   * @param cutoff        EAP cutoff; Attempt to prune alignments with cost > cutoff. May lead to early abandoning.
   *                      ub = PINF: Pruning using the cost of an arbitrary alignment
   *                      ub = QNAN: No cutoff: no pruning nor early abandoning
   *                      ub = other value: use for pruning and early abandoning
   * @param buffers_v     Buffer used to perform the computation. Will reallocate if required.
   * @return ADTW between the two series or +INF if early abandoned.
   */
  template<typename F>
  inline F adtw(size_t length1,
                size_t length2,
                utils::ICFun<F> auto cfun,
                F penalty,
                F cutoff,
                std::vector<F>& buffer_v
  ) {
    constexpr F PINF = utils::PINF<F>;

    if (length1==0&&length2==0) { return 0; }
    else if ((length1==0)!=(length2==0)) { return PINF; }
    else {
      // Compute a cutoff point using the diagonal
      if (std::isinf(cutoff)) {
        cutoff = 0;
        // Cover diagonal
        const auto m = std::min(length1, length2);
        for (size_t i{0}; i<m; ++i) { cutoff = cutoff + cfun(i, i); }
        // Fewer line than columns: complete the last line (advance in the columns)
        if (length1<length2) {
          for (size_t j{length1}; j<length2; ++j) { cutoff = cutoff + cfun(length1 - 1, j) + penalty; }
        } // Fewer columns than lines: complete the last column (advance in the lines)
        else if (length2<length1) {
          for (size_t i{length2}; i<length1; ++i) { cutoff = cutoff + cfun(i, length2 - 1) + penalty; }
        }
      } else if (std::isnan(cutoff)) { cutoff = PINF; }
      // ub computed
      return internal::adtw<F>(length1, length2, cfun, penalty, cutoff, buffer_v);
    }
  }

  /// Helper for the above without having to provide a buffer
  template<typename F>
  inline F adtw(size_t length1, size_t length2, utils::ICFun<F> auto cfun, F penalty, F cutoff) {
    std::vector<F> v;
    return adtw<F>(length1, length2, cfun, penalty, cutoff, v);
  }

} // End of namespace tempo::distance