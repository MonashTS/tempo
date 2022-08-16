#pragma once

#include "../utils.private.hpp"

namespace tempo::distance {

  namespace internal {

    /** Unconstrained (no window) Dynamic Time Warping, Early Abandoned and Pruned (EAP).
     * @tparam F            Floating type used for the computation
     * @param nblines       Length of the line series.
     * @param nbcols        Length of the column series.
     * @param cfun          Indexed Cost function between two points
     * @param cutoff        Attempt to prune computation of alignments with cost > cutoff.
     *                      May lead to early abandoning.
     * @param buffer_v      The buffer used to carry the computation.
     * @return DTW between the two series or +INF if early abandoned.
     */
    template<typename F>
    F dtw(const size_t nblines,
          const size_t nbcols,
          utils::ICFun<F> auto cfun,
          const F cutoff,
          std::vector<F> buffer_v
    ) {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // in debug mode, check preconditions
      assert(nblines!=0);
      assert(nbcols!=0);

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Adapt constants to the floating point type
      using utils::min;
      constexpr F PINF = utils::PINF<F>;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
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
        // Check against the *original cutoff* dealing with the case where we have both series of length 1.
        cost = cfun(0, 0);
        if (cost>cutoff) { return PINF; }
        buffer[c + 0] = cost;
        // All other cells. Checking against "ub" is OK as the only case where the last cell of this line is the
        // last alignment is taken are just above (1==nblines==nbcols, and we have nblines >= nbcols).
        size_t curr_pp = 1;
        for (j = 1; j==curr_pp&&j<nbcols; ++j) {
          cost = cost + cfun(0, j);
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
          cost = buffer[p + j] + cfun(i, j);
          buffer[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
          ++j;
        }
        // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        for (; j==next_start&&j<prev_pp; ++j) {
          cost = std::min(buffer[p + j - 1], buffer[p + j]) + cfun(i, j);
          buffer[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
        }
        // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        for (; j<prev_pp; ++j) {
          cost = min(cost, buffer[p + j - 1], buffer[p + j]) + cfun(i, j);
          buffer[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; }
        }
        // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if (j<nbcols) { // If so, two cases.
          if (j==next_start) { // Case 1: Advancing next start: only diag.
            cost = buffer[p + j - 1] + cfun(i, j);
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
            else {
              // Special case if we are on the last alignment: return the actual cost if we are <= *original cutoff*
              if (i==nblines - 1&&j==nbcols - 1&&cost<=cutoff) { return cost; } else { return PINF; }
            }
          } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
            cost = std::min(cost, buffer[p + j - 1]) + cfun(i, j);
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
          }
          ++j;
        } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
          if (j==next_start) {
            // But only if we are above the *original cutoff*
            // Else set the next starting point to the last valid column
            if (cost>cutoff) { return PINF; } else { next_start = nbcols - 1; }
          }
        }
        // --- --- --- Stage 4: After the previous pruning point: only prev.
        // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
        for (; j==curr_pp&&j<nbcols; ++j) {
          cost = cost + cfun(i, j);
          buffer[c + j] = cost;
          if (cost<=ub) { ++curr_pp; }
        }
        // --- --- ---
        prev_pp = curr_pp;
      } // End of main loop for(;i<nblines;++i)

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Finalisation
      // Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols).
      // Cost must be <= *original cutoff*.
      if (j==nbcols&&cost<=cutoff) { return cost; } else { return PINF; }
    }

    /** Dynamic Time Warping with warping window, Early Abandoned and Pruned (EAP).
     * @tparam F            Floating type used for the computation
     * @param nblines       Length of the line series.
     * @param nbcols        Length of the column series.
     * @param cfun          Indexed Cost function between two points
     * @param window        Warping window
     * @param cutoff        Attempt to prune computation of alignments with cost > cutoff.
     *                      May lead to early abandoning.
     * @param buffer_v      The buffer used to carry the computation.
     * @return DTW between the two series or +INF if early abandoned.
     */
    template<typename F>
    F dtw(const size_t nblines,
          const size_t nbcols,
          utils::ICFun<F> auto cfun,
          const size_t window,
          const F cutoff,
          std::vector<F>& buffer_v
    ) {
      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // In debug mode, check preconditions
      assert(nblines!=0);
      assert(nbcols!=0);

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Adapt constants to the floating point type
      using namespace utils;
      constexpr F PINF = utils::PINF<F>;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Create a new tighter upper bounds (most commonly used in the code).
      // First, take the "next float" after "cutoff" to deal with numerical instability.
      // Then, subtract the cost of the last alignment.
      const F ub = nextafter(cutoff, PINF) - cfun(nblines - 1, nbcols - 1);

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
      // Initialisation of the top border: already initialized to +INF. Initialise the left corner to 0.
      buffer[c - 1] = 0;

      // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
      // Main loop
      for (; i<nblines; ++i) {
        // --- --- --- Swap and variables init
        std::swap(c, p);
        const size_t jStart = std::max(cap_start_index_to_window(i, window), next_start);
        const size_t jStop = cap_stop_index_to_window_or_end(i, window, nbcols);
        next_start = jStart;
        size_t curr_pp = next_start; // Next pruning point init at the start of the line
        j = next_start;
        // --- --- --- Stage 0: Initialise the left border
        {
          cost = PINF;
          buffer[c + jStart - 1] = cost;
        }
        // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
        for (; j==next_start&&j<prev_pp; ++j) {
          const auto d = cfun(i, j);
          cost = std::min(buffer[p + j - 1], buffer[p + j]) + d;
          buffer[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
        }
        // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
        for (; j<prev_pp; ++j) {
          const auto d = cfun(i, j);
          cost = min(cost, buffer[p + j - 1], buffer[p + j]) + d;
          buffer[c + j] = cost;
          if (cost<=ub) { curr_pp = j + 1; }
        }
        // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
        if (j<jStop) { // If so, two cases.
          const auto d = cfun(i, j);
          if (j==next_start) { // Case 1: Advancing next start: only diag.
            cost = buffer[p + j - 1] + d;
            buffer[c + j] = cost;
            if (cost<=ub) { curr_pp = j + 1; }
            else {
              // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
              if (i==nblines - 1&&j==nbcols - 1&&cost<=cutoff) { return cost; }
              else { return PINF; }
            }
          } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
            cost = std::min(cost, buffer[p + j - 1]) + d;
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
        for (; j==curr_pp&&j<jStop; ++j) {
          const auto d = cfun(i, j);
          cost = cost + d;
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

  /** Dynamic Time Warping (DTW), Early Abandoned and Pruned (EAP)
   * @tparam F          Floating type used for the computation
   * @param length1     Length of the first series.
   * @param length2     Length of the second series.
   * @param cfun        Indexed Cost function between two points
   * @param window      Warping window length - align point within 'w' (look 'w' cells on each side of the diagonal)
   *                    Having w > L-2 is the same as having no window (where L = max length)
   * @param cutoff      EAP cutoff; Attempt to prune alignments with cost > cutoff. May lead to early abandoning.
   *                    ub = PINF: Pruning using the cost of an arbitrary alignment
   *                    ub = QNAN: No cutoff: no pruning nor early abandoning
   *                    ub = other value: use for pruning and early abandoning
   * @param buffers_v   Buffer used to perform the computation. Will reallocate if required.
   * @return DTW between the two series or +INF if early abandoned.
   */
  template<typename F>
  inline F dtw(size_t length1,
               size_t length2,
               utils::ICFun<F> auto cfun,
               size_t window,
               F cutoff,
               std::vector<F>& buffer_v
  ) {
    constexpr F PINF = utils::PINF<F>;
    if (length1==0&&length2==0) { return 0; }
    else if ((length1==0)!=(length2==0)) { return PINF; }
    else {
      // Check that the window allows for an alignment
      const auto m = std::min(length1, length2);
      const auto M = std::max(length1, length2);
      if (M - m>window) { return PINF; }
      // Compute a cutoff point using the diagonal - window is valid is large enough to take side steps
      if (std::isinf(cutoff)) {
        cutoff = 0;
        // Cover diagonal
        for (size_t i{0}; i<m; ++i) { cutoff = cutoff + cfun(i, i); }
        // Fewer line than columns: complete the last line
        if (length1<length2) { for (size_t i{length1}; i<length2; ++i) { cutoff = cutoff + cfun(length1 - 1, i); }}
          // Fewer columns than lines: complete the last column
        else if (length2<length1) { for (size_t i{length2}; i<length1; ++i) { cutoff = cutoff + cfun(i, length2 - 1); }}
      } else if (std::isnan(cutoff)) { cutoff = PINF; }
      // ub computed: choose the version to call
      if (window>M - 2) { return internal::dtw(length1, length2, cfun, cutoff, buffer_v); }
      else { return internal::dtw(length1, length2, cfun, window, cutoff, buffer_v); }
    }
  }

  /// Helper for the above without having to provide a buffer
  template<typename F>
  inline F dtw(size_t length1, size_t length2, utils::ICFun<F> auto cfun, size_t window, F cutoff) {
    std::vector<F> v;
    return dtw<F>(length1, length2, cfun, window, cutoff, v);
  }

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // Warping Result version of the code
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  namespace WR {

    namespace internal {

      /** DTW EAP returning a WarpingResult
       *  Detect the max deviation from the diagonal (i.e. the smallest window for which the result is still valid).
       *  Uses an extra buffer to keep track of the max deviation.
       */
      template<typename F>
      tempo::distance::WR::WarpingResult<F> dtw(const size_t nblines,
                                                const size_t nbcols,
                                                utils::ICFun<F> auto cfun,
                                                const size_t window,
                                                const F cutoff,
                                                std::vector<F>& buffer_v,
                                                std::vector<size_t>& buffer_wv
      ) {
        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // In debug mode, check preconditions
        assert(nblines!=0);
        assert(nbcols!=0);

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Adapt constants to the floating point type
        using namespace utils;
        constexpr F PINF = utils::PINF<F>;

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Create a new tighter upper bounds (most commonly used in the code).
        // First, take the "next float" after "cutoff" to deal with numerical instability.
        // Then, subtract the cost of the last alignment.
        const F ub = nextafter(cutoff, PINF) - cfun(nblines - 1, nbcols - 1);

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

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Window validity (marked WV in the code)
        // Double buffer used to compute the window validity
        buffer_wv.assign((1 + nbcols)*2, 0);
        auto *buffer_validity = buffer_wv.data();
        // Window's bound
        size_t mw{0};

        // EAP variables: track where to start the next line, and the position of the previous pruning point.
        // Must be init to 0: index 0 is the next starting index and also the "previous pruning point"
        size_t next_start{0}, prev_pp{0};

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Initialisation of the top border: already initialised to +INF. Initialise the left corner to 0.
        buffer[c - 1] = 0;
        // buffer_validity[c-1] = 0; // WV - already done

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Main loop
        for (; i<nblines; ++i) {
          // --- --- --- Swap and variables init
          std::swap(c, p);
          const size_t jStart = std::max(cap_start_index_to_window(i, window), next_start);
          const size_t jStop = cap_stop_index_to_window_or_end(i, window, nbcols);
          next_start = jStart;
          size_t curr_pp = next_start; // Next pruning point init at the start of the line
          j = next_start;
          // --- --- --- Stage 0: Initialise the left border
          {
            cost = PINF;
            buffer[c + jStart - 1] = cost;
          }
          // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
          for (; j==next_start&&j<prev_pp; ++j) {
            const auto d = cfun(i, j);
            size_t dw = utils::absdiff(i, j);   // WV
            if (buffer[p + j - 1]<=buffer[p + j]) {
              // DIAG
              cost = buffer[p + j - 1] + d;
              mw = buffer_validity[p + j - 1]; // WV
            } else {
              // TOP
              cost = buffer[p + j] + d;
              mw = std::max(dw, buffer_validity[p + j]); // WV
            }
            buffer[c + j] = cost;
            buffer_validity[c + j] = mw;  // WV
            if (cost<=ub) { curr_pp = j + 1; } else { ++next_start; }
          }
          // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
          for (; j<prev_pp; ++j) {
            const auto d = cfun(i, j);
            size_t dw = utils::absdiff(i, j);   // WV
            if (buffer[p + j - 1]<=cost&&buffer[p + j - 1]<=buffer[p + j]) {
              // DIAG -- do diagonal first in case of tie
              cost = buffer[p + j - 1] + d;
              mw = buffer_validity[p + j - 1];     // WV
            } else if (cost<=buffer[p + j - 1]&&cost<=buffer[p + j]) {
              // LEFT
              cost = cost + d;
              mw = std::max(dw, buffer_validity[c + j - 1]); // WV
            } else {
              // TOP
              assert (buffer[p + j]<=cost&&buffer[p + j]<=buffer[p + j - 1]);
              cost = buffer[p + j] + d;
              mw = std::max(dw, buffer_validity[p + j]);  // WV
            }
            buffer[c + j] = cost;
            buffer_validity[c + j] = mw; // WV
            if (cost<=ub) { curr_pp = j + 1; }
          }
          // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
          if (j<jStop) { // If so, two cases.
            const auto d = cfun(i, j);
            size_t dw = utils::absdiff(i, j);   // WV
            if (j==next_start) { // Case 1: Advancing next start: only diag.
              // DIAG
              cost = buffer[p + j - 1] + d;
              mw = buffer_validity[p + j - 1]; // WV
              buffer[c + j] = cost;
              buffer_validity[c + j] = mw;    // WV
              if (cost<=ub) { curr_pp = j + 1; }
              else {
                // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
                if (i==nblines - 1&&j==nbcols - 1&&cost<=cutoff) { return WarpingResult(cost, mw); }
                else { return WarpingResult<F>(); }
              }
            } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
              if (cost<=buffer[p + j - 1]) {
                // LEFT
                cost = cost + d;
                mw = std::max(dw, buffer_validity[c + j - 1]); // WV
              } else {
                // DIAG
                cost = buffer[p + j - 1] + d;
                mw = buffer_validity[p + j - 1];    // WV
              }
              buffer[c + j] = cost;
              buffer_validity[c + j] = mw;    // WV
              if (cost<=ub) { curr_pp = j + 1; }
            }
            ++j;
          } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
            if (j==next_start) {
              // But only if we are above the original UB
              // Else set the next starting point to the last valid column
              if (cost>cutoff) { return WarpingResult<F>(); } else { next_start = nbcols - 1; }
            }
          }
          // --- --- --- Stage 4: After the previous pruning point: only prev.
          // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
          for (; j==curr_pp&&j<jStop; ++j) {
            const auto d = cfun(i, j);
            cost = cost + d;
            buffer[c + j] = cost;

            size_t dw = utils::absdiff(i, j);   // WV
            mw = std::max(dw, buffer_validity[c + j - 1]); // WV: Assess window when moving horizontally
            buffer_validity[c + j] = mw;

            if (cost<=ub) { ++curr_pp; }
          }
          // --- --- ---
          prev_pp = curr_pp;
        } // End of main loop for(;i<nblines;++i)

        // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        // Finalisation
        // Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
        if (j==nbcols&&cost<=cutoff) { return WarpingResult<F>(cost, mw); } else { return WarpingResult<F>(); }
      }

    } // End of namespace internal

    /** DTW EAP returning a WarpingResult
     *  Detect the max deviation from the diagonal (i.e. the smallest window for which the result is still valid).
     *  Uses an extra buffer to keep track of the max deviation.
     */
    template<typename F>
    inline WarpingResult<F> dtw(size_t nblines,
                                size_t nbcols,
                                utils::ICFun<F> auto cfun,
                                size_t window,
                                F cutoff,
                                std::vector<F>& buffer_v,
                                std::vector<size_t>& buffer_wv
    ) {
      using utils::PINF;
      if (nblines==0&&nbcols==0) { return WarpingResult<F>(0, 0); }
      else if ((nblines==0)!=(nbcols==0)) { return WarpingResult<F>(); }
      else {
        // Check that the window allows for an alignment
        // If this is accepted, we do not need to check the window when computing a new UB
        const auto m = std::min(nblines, nbcols);
        const auto M = std::max(nblines, nbcols);
        if (M - m>window) { return WarpingResult<F>(); }
        // Compute a cutoff point using the diagonal
        if (std::isinf(cutoff)) {
          cutoff = 0;
          // Cover diagonal
          for (size_t i{0}; i<m; ++i) { cutoff = cutoff + cfun(i, i); }
          // Fewer line than columns: complete the last line
          if (nblines<nbcols) { for (size_t i{nblines}; i<nbcols; ++i) { cutoff = cutoff + cfun(nblines - 1, i); }}
            // Fewer columns than lines: complete the last column
          else if (nbcols<nblines) { for (size_t i{nbcols}; i<nblines; ++i) { cutoff = cutoff + cfun(i, nbcols - 1); }}
        } else if (std::isnan(cutoff)) { cutoff = PINF<F>; }
        // ub computed
        return internal::dtw(nblines, nbcols, cfun, window, cutoff, buffer_v, buffer_wv);
      }
    }

    /// Helper without having to provide buffers
    template<typename F>
    inline WarpingResult<F> dtw(size_t nblines, size_t nbcols, utils::ICFun<F> auto dist, size_t w, F ub) {
      std::vector<F> v;
      std::vector<size_t> wv;
      return dtw(nblines, nbcols, dist, w, ub, v, wv);
    }

  } // End of namespace WR


} // End of namespace tempo::distance
