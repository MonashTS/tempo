#pragma once

#include "../utils.private.hpp"

namespace tempo::distance {

  namespace internal {

    /** Time Warp Edit (TWE), Early Abandoned and Pruned (EAP).
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
     * @return TWE value or +INF if early abandoned, or , given w, no alignment is possible
     */
    template<typename F>
    F twe(const size_t nblines,
          const size_t nbcols,
          utils::ICFunOne<F> auto cfun_lines,
          utils::ICFunOne<F> auto cfun_cols,
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

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      const F ub = [&]() -> F {
        // The last alignment can only computed if we have nbcols >= 2
        if (nbcols>=2) {
          const auto la = min(
            // "Delete_B": over the columns / Prev
            // dist(cols, nbcols-2, cols, nbcols-1)+nu_lambda --> Capture in dist_cols
            cfun_cols(nbcols - 1),
            // Match: Diag. Ok: nblines >= nbcols
            // dist(lines, nblines-1, cols, nbcols-1)+dist(lines, nblines-2, cols, nbcols-2)+nu2d(nblines-nbcols),
            cfun_diag(nblines - 1, nbcols - 1),
            // "Delete_A": over the lines / Top
            // dist(lines, nblines-2, lines, nblines-1)+nu_lambda --> Capture in dist_lines
            cfun_lines(nblines - 1)
          );
          return F(nextafter(cutoff, PINF) - la);
        } else { return F(cutoff); }
      }();

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Double buffer allocation, no initialisation required (border condition manage in the code).
      // Base indices for the 'c'urrent row and the 'p'revious row.
      // Also allocate for "distcol", buffer holding precomputed distance between columns (of length nbcols)
      buffer_v.assign(nbcols*3, 0); // 3*nbcols : 2 for row + 1 for distcol

      F *buffers = buffer_v.data();
      size_t c{0}, p{nbcols};

      F *distcol = buffers + (nbcols*2);

      // Line & column counters
      size_t i{0}, j{0};

      // Cost accumulator. Also used as the "left neighbour".
      F cost{0};

      // EAP variables: track where to start the next line, and the position of the previous pruning point.
      // Must be init to 0: index 0 is the next starting index and also the "previous pruning point"
      size_t next_start{0}, prev_pp{0};

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Initialisation of the first line. Deal with the line top border condition.
      {
        // Case [0,0]: special "Match case"
        cost = cfun_diag(0, 0);
        buffers[c + 0] = cost;
        // Distance for the first column is relative to 0 "by conventions" (from the paper, section 4.2)
        // Something like
        // distcol[0]=dist(0, cols[0]);
        // However, this would only be use for j=0, i.e. with the left border which is +INF,
        // hence the result would always be +INF: we can simply use 0 instead.
        // Note that the border is managed as part of the code, hence we actually never access distcol[0]!
        // distcol[0] = 0;
        // Rest of the line: [i==0, j>=1]: "Delete_B case" (prev)
        // We also initialize 'distcol' here.
        for (j = 1; j<nbcols; ++j) {
          const F d = cfun_cols(j);
          distcol[j] = d;
          cost = cost + d;
          buffers[c + j] = cost;
          if (cost<=ub) { prev_pp = j + 1; } else { break; }
        }
        // Complete the initialisation of distcol
        for (; j<nbcols; ++j) { distcol[j] = cfun_cols(j); }
        // Next line.
        ++i;
      }


      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Main loop, starts at the second line
      for (; i<nblines; ++i) {
        // --- --- --- Swap and variables init
        std::swap(c, p);
        const F distli = cfun_lines(i);
        size_t curr_pp = next_start; // Next pruning point init at the start of the line
        j = next_start;
        // --- --- --- Stage 0: Special case for the first column. Can only look up (border on the left)
        {
          cost = buffers[p + j] + distli; // "Delete_A" / Top
          buffers[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
          ++j;
        }
        // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        for (; j==next_start&&j<prev_pp; ++j) {
          cost = std::min(
            buffers[p + j - 1] + cfun_diag(i, j),     // "Match" / Diag
            buffers[p + j] + distli                   // "Delete_A" / Top
          );
          buffers[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
        }
        // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        for (; j<prev_pp; ++j) {
          cost = min(
            cost + distcol[j],                      // "Delete_B": over the columns / Prev
            buffers[p + j - 1] + cfun_diag(i, j),   // Match: Diag
            buffers[p + j] + distli                 // "Delete_A": over the lines / Top
          );
          buffers[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; }
        }
        // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if (j<nbcols) { // If so, two cases.
          if (j==next_start) { // Case 1: Advancing next start: only diag.
            cost = buffers[p + j - 1] + cfun_diag(i, j); // Match: Diag
            buffers[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
            else {
              // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
              if (i==nblines - 1&&j==nbcols - 1&&cost<=cutoff) { return cost; } else { return PINF; }
            }
          } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
            cost = std::min(
              cost + distcol[j],                      // "Delete_B": over the columns / Prev
              buffers[p + j - 1] + cfun_diag(i, j)    // Match: Diag
            );
            buffers[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
          }
          ++j;
        } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
          if (j==next_start) {
            // But only if we are above the original UB. Else set the next starting point to the last valid column
            if (cost>cutoff) { return PINF; } else { next_start = nbcols - 1; }
          }
        }
        // --- --- --- Stage 4: After the previous pruning point: only prev.
        // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
        for (; j==curr_pp&&j<nbcols; ++j) {
          cost = cost + distcol[j]; // "Delete_B": over the columns / Prev
          buffers[c + j] = cost;
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


  /** Time Warp Edit (TWE), Early Abandoned and Pruned (EAP).
   *
   * @tparam F              Floating type used for the computation
   * @param length1         Length of the first series.
   * @param length2         Length of the second series.
   * @param cfun_lines      Indexed Cost Function for "vertical" steps
   * @param cfun_cols       Indexed Cost Function for "horizontal" steps
   * @param cfun_diag       Indexed Cost Function for "diagonal" steps
   * @param cutoff          EAP cutoff; Attempt to prune alignments with cost > cutoff. May lead to early abandoning.
   *                        ub = PINF: Pruning using the cost of an arbitrary alignment
   *                        ub = QNAN: No cutoff: no pruning nor early abandoning
   *                        ub = other value: use for pruning and early abandoning
   * @param buffers_v       Buffer used to perform the computation. Will reallocate if required.
   * @return TWE value or +INF if early abandoned, or , given w, no alignment is possible
   */
  template<typename F>
  inline F twe(const size_t length1,
               const size_t length2,
               utils::ICFunOne<F> auto cfun_lines,
               utils::ICFunOne<F> auto cfun_cols,
               utils::ICFun<F> auto cfun_diag,
               F cutoff,
               std::vector<F>& buffer_v
  ) {
    constexpr F PINF = utils::PINF<F>;
    if (length1==0&&length2==0) { return 0; }
    else if ((length1==0)!=(length2==0)) { return PINF; }
    else {
      // Compute a cutoff point using the diagonal
      if (std::isinf(cutoff)) {
        const auto m = std::min(length1, length2);
        cutoff = 0;
        // Init case
        cutoff = cfun_diag(0, 0);
        // Cover diagonal
        for (size_t i{1}; i<m; ++i) { cutoff = cutoff + cfun_diag(i, i); }
        // Fewer line than columns: complete the last line (advancing in the columns)
        if (length1<length2) {
          for (size_t j{length1}; j<length2; ++j) { cutoff = cutoff + cfun_cols(j); }
        }
          // Fewer columns than lines: complete the last column (advancing in the lines)
        else if (length2<length1) {
          for (size_t i{length2}; i<length1; ++i) { cutoff = cutoff + cfun_lines(i); }
        }
      } else if (std::isnan(cutoff)) { cutoff = PINF; }
      // ub computed
      return internal::twe(length1, length2, cfun_lines, cfun_cols, cfun_diag, cutoff, buffer_v);
    }
  }

  /// Helper without having to provide a buffer
  template<typename F>
  inline F twe(const size_t length1,
               const size_t length2,
               utils::ICFunOne<F> auto cfun_lines,
               utils::ICFunOne<F> auto cfun_cols,
               utils::ICFun<F> auto cfun_diag,
               F cutoff
  ) {
    std::vector<F> v;
    const F r = twe(length1, length2, cfun_lines, cfun_cols, cfun_diag, cutoff, v);
    return r;
  }


  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Specific cost functions
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

  namespace univariate {

    /// Default TWE warping step cost function (ad2) - Indexed Cost Function (One) Builder
    /// Note: Consider timestamp spaced by a const unit of 1.
    ///       Warp case: we always have a time difference of 1, so we always add 1*nu+lambda
    /// @param nu            Stiffness parameter
    /// @param lambda        Penalty parameter
    template<typename F, utils::Subscriptable D>
    inline utils::ICFunOne<F> auto idx_twe_warp(D const& s, const F nu, const F lambda) {
      const F nl = nu + lambda;
      return [&, nl](size_t i) { return ad2(s[i], s[i - 1]) + nl; };
    }

    /// Default TWE diagonal step cost function (ad2) - Indexed Cost Function Builder
    /// Note: Consider timestamp spaced by a const unit of 1.
    ///       Match case: we always have nu*(|i-j|+|(i-1)-(j-1)|) == 2*nu*|i-j|
    /// @param nu            Stiffness parameter
    template<typename F, utils::Subscriptable D>
    inline utils::ICFun<F> auto idx_twe_match(D const& s1, D const& s2, const F nu) {
      const auto nu2 = F(2)*nu;
      return [&, nu2](size_t i, size_t j) {
        const F da = ad2(s1[i], s2[j]);
        const F db = ad2(s1[i - 1], s2[j - 1]);
        return da + db + nu2*utils::absdiff(i, j);
      };
    }

  } // End of namespace univariate


  namespace multivariate {

  } // End of namespace multivariate

} // End of namespace tempo::distance
