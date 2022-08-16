#pragma once

#include "../utils.private.hpp"

namespace tempo::distance {

  namespace internal {

    /** Move Split Merge (MSM), Early Abandoned and Pruned (EAP).
     *
     * @tparam F            Floating type used for the computation
     * @param nblines       Length of the first series 'along the lines'
     * @param nbcols        Length of the second series 'along the columnes'
     * @param cfun_lines    Indexed Cost Function for "vertical" steps
     * @param cfun_cols     Indexed Cost Function for "horizontal" steps
     * @param cfun_diag     Indexed Cost Function for "diagonal" steps
     * @param cutoff        EAP cutoff; Attempt to prune alignments with cost > cutoff. May lead to early abandoning.
     *                      ub = PINF: Pruning using the cost of an arbitrary alignment
     *                      ub = QNAN: No cutoff: no pruning nor early abandoning
     *                      ub = other value: use for pruning and early abandoning
     * @param buffers_v     Buffer used to perform the computation. Will reallocate if required.
     * @return MSM value or +INF if early abandoned, or , given w, no alignment is possible
     */
    template<typename F>
    F msm(const size_t nblines,
          const size_t nbcols,
          utils::ICFun<F> auto cfun_lines,
          utils::ICFun<F> auto cfun_cols,
          utils::ICFun<F> auto cfun_diag,
          F cutoff,
          std::vector<F>& buffer_v
    ) {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // In debug mode, check preconditions
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
        // The last alignment can only computed if we have nbcols >= 2
        if (nbcols>=2) {
          const auto i = nblines - 1;
          const auto j = nbcols - 1;
          const auto la = min(
            cfun_diag(i, j),  // Diag: Move
            cfun_cols(i, j),  // Previous: Split/Merge
            cfun_lines(i, j)  // Above: Split/Merge
          );
          return F(nextafter(cutoff, PINF - la));
        } else {
          return F(cutoff); // Force type to prevent auto-deduction failure
        }
      }();


      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Double buffer allocation, no initialisation required (border condition manage in the code).
      // Base indices for the 'c'urrent row and the 'p'revious row.
      buffer_v.assign((nbcols*2), 0);
      auto *buffer = buffer_v.data();
      size_t c{0}, p{nbcols};

      // Line & column counters
      size_t i{0}, j{0};

      // Cost accumulator. Also used as the "left neighbour".
      F cost{0};

      // EAP variables: track where to start the next line, and the position of the previous pruning point.
      // Must be init to 0: index 0 is the next starting index and also the "previous pruning point"
      size_t next_start{0}, prev_pp{0};

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Initialisation: compute the first line. Required as the main loop starts at line=1, not 0.
      {
        // First cell (0,0) is a special case. Early abandon if above the cut-off point.
        {
          cost = cfun_diag(0, 0); // Very first cell
          buffer[c + 0] = cost;
          if (cost<=ub) { prev_pp = 1; } else { return PINF; }
        }
        // Rest of the line, a cell only depends on the previous cell. Stop when > ub, update prev_pp.
        for (j = 1; j<nbcols; ++j) {
          cost = cost + cfun_cols(0, j); // Previous
          buffer[c + j] = cost;
          if (cost<=ub) { prev_pp = j + 1; } else { break; }
        }
        // Next line.
        ++i;
      }

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Main loop
      for (; i<nblines; ++i) {
        // --- --- --- Swap and variables init
        std::swap(c, p);
        size_t curr_pp = next_start; // Next pruning point init at the start of the line
        j = next_start;
        // --- --- --- Stage 0: Special case for the first column. Can only look up (border on the left)
        {
          cost = buffer[p + j] + cfun_lines(i, j);  // Above
          buffer[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
          ++j;
        }
        // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        for (; j==next_start&&j<prev_pp; ++j) {
          cost = std::min(
            buffer[p + j - 1] + cfun_diag(i, j),            // Diag: Move
            buffer[p + j] + cfun_lines(i, j)                // Above: Split/Merge
          );
          buffer[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
        }
        // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        for (; j<prev_pp; ++j) {
          cost = min(
            buffer[p + j - 1] + cfun_diag(i, j),      // Diag: Move
            cost + cfun_cols(i, j),                   // Previous: Split/Merge
            buffer[p + j] + cfun_lines(i, j)          // Above: Split/Merge
          );
          buffer[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; }
        }
        // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if (j<nbcols) { // If so, two cases.
          if (j==next_start) { // Case 1: Advancing next start: only diag.
            cost = buffer[p + j - 1] + cfun_diag(i, j);    // Diag: Move
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
            else {
              // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
              if (i==nblines - 1&&j==nbcols - 1&&cost<=cutoff) { return cost; }
              else { return PINF; }
            }
          } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
            cost = std::min(
              buffer[p + j - 1] + cfun_diag(i, j),    // Diag: Move
              cost + cfun_cols(i, j)                  // Previous: Split/Merge
            );
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
          }
          ++j;
        } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
          if (j==next_start) {
            // But only if we are above the original UB
            // Else set the next starting point to the last valid column
            if (cost>cutoff) { return PINF; }
            else { next_start = nbcols - 1; }
          }
        }
        // --- --- --- Stage 4: After the previous pruning point: only prev.
        // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
        for (; j==curr_pp&&j<nbcols; ++j) {
          cost = cost + cfun_cols(i, j);    // Previous: Split/Merge
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



  /** Move Split Merge (MSM),  Early Abandoned and Pruned (EAP).
   *
   * @tparam F            Floating type used for the computation
   * @param length1       Length of the first series
   * @param length2       Length of the second series
   * @param cfun_lines    Indexed Cost Function for "vertical" steps
   * @param cfun_cols     Indexed Cost Function for "horizontal" steps
   * @param cfun_diag     Indexed Cost Function for "diagonal" steps
   * @param window        Warping window length - align point within 'w' (look 'w' cells on each side of the diagonal)
   *                      Having w > L-2 is the same as having no window (where L = max length)
   * @param cutoff        EAP cutoff; Attempt to prune alignments with cost > cutoff. May lead to early abandoning.
   *                      ub = PINF: Pruning using the cost of an arbitrary alignment
   *                      ub = QNAN: No cutoff: no pruning nor early abandoning
   *                      ub = other value: use for pruning and early abandoning
   * @param buffers_v     Buffer used to perform the computation. Will reallocate if required.
   * @return MSM value or +INF if early abandoned, or , given w, no alignment is possible
   */
  template<typename F>
  inline F msm(const size_t nblines,
               const size_t nbcols,
               utils::ICFun<F> auto dist_lines,
               utils::ICFun<F> auto dist_cols,
               utils::ICFun<F> auto dist,
               F cutoff,
               std::vector<F>& buffer_v
  ) {
    constexpr F PINF = utils::PINF<F>;
    if (nblines==0&&nbcols==0) { return 0; }
    else if ((nblines==0)!=(nbcols==0)) { return PINF; }
    else {
      // Compute a cutoff point using the diagonal
      if (std::isinf(cutoff)) {
        const auto m = std::min(nblines, nbcols);
        cutoff = 0;
        // Cover diagonal
        for (size_t i{0}; i<m; ++i) { cutoff = cutoff + dist(i, i); }
        // Fewer line than columns: complete the last line (advancing in the columns)
        if (nblines<nbcols) {
          for (size_t j{nblines}; j<nbcols; ++j) { cutoff = cutoff + dist_cols(nblines - 1, j); }
        }
          // Fewer columns than lines: complete the last column (advancing in the lines)
        else if (nbcols<nblines) {
          for (size_t i{nbcols}; i<nblines; ++i) { cutoff = cutoff + dist_lines(i, nbcols - 1); }
        }
      } else if (std::isnan(cutoff)) { cutoff = PINF; }
      // ub computed
      return internal::msm(nblines, nbcols, dist_lines, dist_cols, dist, cutoff, buffer_v);
    }
  }

  /// Helper without having to provide a buffer
  template<typename F>
  inline F msm(const size_t length1,
               const size_t length2,
               utils::ICFun<F> auto cfun_lines,
               utils::ICFun<F> auto cfun_cols,
               utils::ICFun<F> auto cfun_diag,
               F cutoff) {
    std::vector<F> v;
    return msm(length1, length2, cfun_lines, cfun_cols, cfun_diag, cutoff, v);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Specific cost functions
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  namespace univariate {

    /** Univariate cost function used when transforming X=(x1, x2, ... xi) into Y = (y1, ..., yj) by Split or Merge (symmetric)
     * @tparam FT    The floating number type used to represent the series.
     * @tparam D            Type of underlying collection - given to dist
     * @param X             Main series: the series where a new point is added (can be line or column!)
     * @param xnew_         In either X or Y
     * @param xi_           Last point of X
     * @param Y             The other series
     * @param yj_           Last point of Y
     * @param c             cost of split and merge operation
     * @return MSM cost of the xi-yj alignment (without "recursive" part)
     */
    template<typename F, utils::Subscriptable D>
    inline F _msm_cost(const D& X, size_t xnew_, size_t xi_, const D& Y, size_t yj_, F cost) {
      F xnew = X[xnew_];
      F xi = X[xi_];
      F yj = Y[yj_];
      if (((xi<=xnew)&&(xnew<=yj))||((yj<=xnew)&&(xnew<=xi))) { return cost; }
      else { return cost + std::min(std::abs(xnew - xi), std::abs(xnew - yj)); }
    }

    /// MSM lines Indexed Cost Function Builder
    template<typename F, utils::Subscriptable D>
    inline utils::ICFun<F> auto idx_msm_lines(const D& lines, const D& cols, const F c) {
      return [&, c](size_t i, size_t j) {
        return _msm_cost(lines, i, i - 1, cols, j, c);
      };
    }

    /// MSM columns Indexed Cost Function Builder
    template<typename F, utils::Subscriptable D>
    inline utils::ICFun<F> auto idx_msm_cols(const D& lines, const D& cols, const F c) {
      return [&, c](size_t i, size_t j) {
        return _msm_cost(cols, j, j - 1, lines, i, c);
      };
    }

    /// MSM diagonal Indexed Cost Function Builder
    template<typename F, utils::Subscriptable D>
    constexpr inline utils::ICFun<F> auto idx_msm_diag(const D& lines, const D& cols) {
      return idx_ad1<F, D>(lines, cols);
    }

  } // End of namespace univariate

  namespace multivariate {

  } // End of namespace multivariate

} // End of namespace tempo::distance
