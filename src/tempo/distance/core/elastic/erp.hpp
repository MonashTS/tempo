#pragma once

#include "../utils.private.hpp"

namespace tempo::distance {

  namespace internal {

    /** Edit Distance with Real Penalty (ERP), Early Abandoned and Pruned (EAP).
     *
     * @tparam F            Floating type used for the computation
     * @param nblines       Length of the first series 'along the lines'
     * @param nbcols        Length of the second series 'along the columnes'
     * @param cfun_gv_lines Indexed Cost function CFunOne - gap value for the 'line series'
     * @param cfun_gv_cols  Indexed Cost function CFunOne - gap value for the 'column series'
     * @param cfun          Indexed Cost function of concept CFun - must capture the series
     * @param window      Warping window length - align point within 'w' (look 'w' cells on each side of the diagonal)
     *                    Having w > L-2 is the same as having no window (where L = max length)
     * @param cutoff      EAP cutoff; Attempt to prune alignments with cost > cutoff. May lead to early abandoning.
     *                    ub = PINF: Pruning using the cost of an arbitrary alignment
     *                    ub = QNAN: No cutoff: no pruning nor early abandoning
     *                    ub = other value: use for pruning and early abandoning
     * @param buffers_v   Buffer used to perform the computation. Will reallocate if required.
     * @return ERP between the two series or +INF if early abandoned.
     */
    template<typename F>
    F erp(const size_t nblines,
          const size_t nbcols,
          utils::ICFunOne<F> auto cfun_gv_lines,
          utils::ICFunOne<F> auto cfun_gv_cols,
          utils::ICFun<F> auto cfun,
          const size_t window,
          const F cutoff,
          std::vector<F>& buffer_v
    ) {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // in debug mode, check preconditions
      assert(nblines!=0);
      assert(nbcols!=0);

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Adapt constants to the floating point type
      using utils::min;
      constexpr F PINF = utils::PINF<F>;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      const F ub = [&]() -> F {
        const F la = min(
          cfun_gv_cols(nbcols - 1),         // Previous (col)
          cfun(nblines - 1, nbcols - 1),    // Diagonal
          cfun_gv_lines(nblines - 1)        // Above    (line)
        );
        return nextafter(cutoff, PINF) - la;
      }();

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Double buffer allocation, init to +INF.
      // Base indices for the 'c'urrent row and the 'p'revious row. Account for the extra cell (+1 and +2)
      buffer_v.assign((1 + nbcols)*2, PINF);
      auto *buffer = buffer_v.data();
      size_t c{0 + 1}, p{nbcols + 2};

      // Line & column counters
      size_t i{0}, j{0};

      // Cost accumulator. Also used as the "left neighbour".
      F cost{0};

      // EAP variables: track where to start the next line, and the position of the previous pruning point.
      // Must be init to 0: index 0 is the next starting index and also the "previous pruning point"
      size_t next_start{0}, prev_pp{0};

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Initialisation of the top border
      {   // Matrix Border - Top diagonal
        buffer[c - 1] = 0;
        // Matrix Border - First line
        const size_t jStop = utils::cap_stop_index_to_window_or_end(0, window, nbcols);
        for (j = 0; buffer[c + j - 1]<=ub&&j<jStop; ++j) {
          buffer[c + j] = buffer[c + j - 1] + cfun_gv_cols(j); // Previous
        }
        // Pruning point set to first +INF value (or out of bound)
        prev_pp = j;
      }

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Part 1: Loop with computed left border.
      {   // The left border has a computed value while it's within the window and its value bv <= ub
        // No "front pruning" (next_start) and no early abandoning can occur while in this loop.
        const size_t iStop = utils::cap_stop_index_to_window_or_end(0, window, nblines);
        for (; i<iStop; ++i) {
          // --- --- --- Variables init
          constexpr size_t jStart = 0;
          const size_t jStop = utils::cap_stop_index_to_window_or_end(i, window, nbcols);
          j = jStart;
          size_t curr_pp = jStart; // Next pruning point init at the start of the line
          // --- --- --- Stage 0: Initialise the left border
          {
            // We haven't swap yet, so the 'top' cell is still indexed by 'c-1'.
            cost = buffer[c - 1] + cfun_gv_lines(i);
            if (cost>ub) { break; }
            else {
              std::swap(c, p);
              buffer[c - 1] = cost;
            }
          }
          // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
          // No stage 1 here.
          // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
          for (; j<prev_pp; ++j) {
            cost = min(
              cost + cfun_gv_cols(j),           // Previous
              buffer[p + j - 1] + cfun(i, j),       // Diagonal
              buffer[p + j] + cfun_gv_lines(i)    // Above
            );
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
          }
          // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
          if (j<jStop) { // Possible path in previous cells: left and diag.
            cost = std::min(
              cost + cfun_gv_cols(j),       // Previous
              buffer[p + j - 1] + cfun(i, j)    // Diagonal
            );
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
            ++j;
          }
          // --- --- --- Stage 4: After the previous pruning point: only prev.
          // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
          for (; j==curr_pp&&j<jStop; ++j) {
            cost = cost + cfun_gv_cols(j);  // Previous
            buffer[c + j] = cost;
            if (cost<=ub) { ++curr_pp; }
          }
          // --- --- ---
          prev_pp = curr_pp;
        }
      }

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Part 2: Loop with +INF left border
      {
        for (; i<nblines; ++i) {
          // --- --- --- Swap and variables init
          std::swap(c, p);
          const size_t jStart = std::max(utils::cap_start_index_to_window(i, window), next_start);
          const size_t jStop = utils::cap_stop_index_to_window_or_end(i, window, nbcols);
          j = jStart;
          next_start = jStart;
          size_t curr_pp = jStart; // Next pruning point init at the start of the line
          // --- --- --- Stage 0: Initialise the left border
          {
            cost = PINF;
            buffer[c + jStart - 1] = cost;
          }
          // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
          for (; j==next_start&&j<prev_pp; ++j) {
            cost = std::min(
              buffer[p + j - 1] + cfun(i, j),       // Diagonal
              buffer[p + j] + cfun_gv_lines(i)    // Above
            );
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
          }
          // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
          for (; j<prev_pp; ++j) {
            cost = min(
              cost + cfun_gv_cols(j),         // Previous
              buffer[p + j - 1] + cfun(i, j),     // Diagonal
              buffer[p + j] + cfun_gv_lines(i)  // Above
            );
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
          }
          // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
          if (j<jStop) { // If so, two cases.
            if (j==next_start) { // Case 1: Advancing next start: only diag.
              cost = buffer[p + j - 1] + cfun(i, j);     // Diagonal
              buffer[c + j] = cost;
              if (cost<=ub) { curr_pp = j + 1; }
              else {
                // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
                if (i==nblines - 1&&j==nbcols - 1&&cost<=cutoff) { return cost; } else { return PINF; }
              }
            } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
              cost = std::min(
                cost + cfun_gv_cols(j),     // Previous
                buffer[p + j - 1] + cfun(i, j)  // Diagonal
              );
              buffer[c + j] = cost;
              if (cost<=ub) { curr_pp = j + 1; }
            }
            ++j;
          } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
            if (j==next_start) {
              // But only if we are above the original UB
              // Else set the next starting point to the last valid column
              if (cost>cutoff) { return PINF; } else { next_start = nbcols - 1; }
            }
          }
          // --- --- --- Stage 4: After the previous pruning point: only prev.
          // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
          for (; j==curr_pp&&j<jStop; ++j) {
            cost = cost + cfun_gv_cols(j);
            buffer[c + j] = cost;
            if (cost<=ub) { ++curr_pp; }
          }
          // --- --- ---
          prev_pp = curr_pp;
        }
      }

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Finalisation
      // Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
      if (j==nbcols&&cost<=cutoff) { return cost; } else { return PINF; }
    }

  } // End of namespace internal


  /** Edit Distance with Real Penalty (ERP), Early Abandoned and Pruned (EAP).
   * @tparam F            Floating type used for the computation
   * @param length1       Length of the first series
   * @param length2       Length of the second series
   * @param cfun_gv_lines Indexed Cost function CFunOne - gap value for the 'line series'
   * @param cfun_gv_cols  Indexed Cost function CFunOne - gap value for the 'column series'
   * @param cfun          Indexed Cost function of concept CFun - must capture the series
   * @param window      Warping window length - align point within 'w' (look 'w' cells on each side of the diagonal)
   *                    Having w > L-2 is the same as having no window (where L = max length)
   * @param cutoff      EAP cutoff; Attempt to prune alignments with cost > cutoff. May lead to early abandoning.
   *                    ub = PINF: Pruning using the cost of an arbitrary alignment
   *                    ub = QNAN: No cutoff: no pruning nor early abandoning
   *                    ub = other value: use for pruning and early abandoning
   * @param buffers_v   Buffer used to perform the computation. Will reallocate if required.
   * @return ERP between the two series or +INF if early abandoned.
   */
  template<typename F>
  inline F erp(const size_t length1,
               const size_t length2,
               utils::ICFunOne<F> auto cfun_gv_lines,
               utils::ICFunOne<F> auto cfun_gv_cols,
               utils::ICFun<F> auto cfun,
               const size_t window,
               F cutoff,
               std::vector<F>& buffer_v
  ) {
    constexpr F PINF = utils::PINF<F>;

    if (length1==0&&length2==0) { return 0; }
    else if ((length1==0)!=(length2==0)) { return PINF; }
    else {
      // Check that the window allows for an alignment
      // If this is accepted, we do not need to check the window when computing a new UB
      const auto m = std::min(length1, length2);
      const auto M = std::max(length1, length2);
      if (M - m>window) { return PINF; }
      // Compute a cutoff point using the diagonal
      if (std::isinf(cutoff)) {
        cutoff = 0;
        // Cover diagonal
        for (size_t i{0}; i<m; ++i) { cutoff = cutoff + cfun(i, i); }
        // Fewer line than columns: complete the last line (advancing in the columns)
        if (length1<length2) { for (size_t i{length1}; i<length2; ++i) { cutoff = cutoff + cfun_gv_cols(i); }}
          // Fewer columns than lines: complete the last column (advancing in the lines)
        else if (length2<length1) { for (size_t i{length2}; i<length1; ++i) { cutoff = cutoff + cfun_gv_lines(i); }}
      } else if (std::isnan(cutoff)) { cutoff = PINF; }
      // ub computed
      return internal::erp(length1, length2, cfun_gv_lines, cfun_gv_cols, cfun, window, cutoff, buffer_v);
    }
  }

  /// Helper without having to provide a buffer
  template<typename F>
  inline F erp(const size_t nblines,
               const size_t nbcols,
               utils::ICFunOne<F> auto cfun_gv_lines,
               utils::ICFunOne<F> auto cfun_gv_cols,
               utils::ICFun<F> auto cfun,
               const size_t window,
               F cutoff
  ) {
    std::vector<F> v;
    return erp<F>(nblines, nbcols, cfun_gv_lines, cfun_gv_cols, cfun, window, cutoff, v);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Specific cost functions
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  namespace univariate {

    /// Gap Value Indexed Cost Function Builder - Absolute Difference exponent 1
    template<typename F, utils::Subscriptable D>
    inline utils::ICFunOne<F> auto idx_gvad1(const D& series, const F gv) {
      return [&, gv](size_t i) {
        return ad1(series[i], gv);
      };
    }

    /// Gap Value Indexed Cost Function Builder - Absolute Difference exponent 2
    template<typename F, utils::Subscriptable D>
    inline utils::ICFunOne<F> auto idx_gvad2(const D& series, const F gv) {
      return [&, gv](size_t i) {
        return ad2<F>(series[i], gv);
      };
    }

    /// Gap Value cost function builder - Absolute Difference exponent e
    template<typename F, utils::Subscriptable D>
    inline auto idx_gvade(const F e) {
      return [e](const D& series, F gv) -> utils::ICFunOne<F> auto {
        return [&, gv, e](size_t i) {
          return ade(series[i], gv, e);
        };
      };
    }

  } // End of namespace univariate

  namespace multivariate {

  } // End of namespace multivariate

} // End of namespace tempo::distance
