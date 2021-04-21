#pragma once

#include "../../../../tseries/tseries.hpp"
#include "../../../../utils/utils.hpp"
#include "../../distances.hpp"

namespace tempo::univariate::fastee {

  template<typename FloatType, auto dist = square_dist < FloatType>>
  [[nodiscard]] inline std::tuple <FloatType, size_t> cdtw(
      const FloatType *lines, size_t nblines,
      const FloatType *cols, size_t nbcols,
      const size_t w,
      FloatType cutoff
  ) {
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // In debug mode, check preconditions
    assert(lines != nullptr && nblines != 0 && nblines < MAX_SERIES_LENGTH);
    assert(cols != nullptr && nbcols != 0 && nbcols < MAX_SERIES_LENGTH);
    assert(nbcols <= nblines);
    assert(w <= nblines);
    assert(nblines - nbcols <= w);
    // Adapt constants to the floating point type
    constexpr auto POSITIVE_INFINITY = tempo::POSITIVE_INFINITY<FloatType>;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Create a new tighter upper bounds (most commonly used in the code).
    // First, take the "next float" after "cutoff" to deal with numerical instability.
    // Then, subtract the cost of the last alignment.
    const FloatType ub = nextafter(cutoff, POSITIVE_INFINITY) - dist(lines[nblines - 1], cols[nbcols - 1]);

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Double buffer allocation, init to +INF. Account for the extra cell (+1 and +2)
    std::vector <FloatType> buffers_v((1 + nbcols) * 2, POSITIVE_INFINITY);
    auto *buffers = buffers_v.data();
    // Double buffer allocation for window size tracking
    std::vector <size_t> buffers_w_v((1 + nbcols) * 2, 0);
    auto *buffers_w = buffers_w_v.data();
    // Base indices for the 'c'urrent row and the 'p'revious row.
    size_t c{0 + 1}, p{nbcols + 2};

    // Line & column counters
    size_t i{0}, j{0};

    // Cost accumulator. Also used as the "left neighbour".
    double cost{0};
    size_t mw{0};   // Window bound

    // EAP variables: track where to start the next line, and the position of the previous pruning point.
    // Must be init to 0: index 0 is the next starting index and also the "previous pruning point"
    size_t next_start{0}, prev_pp{0};

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Initialisation of the top border: already initialized to +INF. Initialise the left corner to 0.
    // Border init to +INF, so the window is not influenced by them (no path from the border itself)
    buffers[c - 1] = 0;
    buffers_w[c - 1] = 0;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Main loop
    for (; i < nblines; ++i) {
      // --- --- --- Swap and variables init
      std::swap(c, p);
      const double li = lines[i];
      const size_t jStart = std::max(cap_start_index_to_window(i, w), next_start);
      const size_t jStop = cap_stop_index_to_window_or_end(i, w, nbcols);
      next_start = jStart;
      size_t curr_pp = next_start; // Next pruning point init at the start of the line
      j = next_start;
      // --- --- --- Stage 0: Initialise the left border
      {
        cost = POSITIVE_INFINITY;
        buffers[c + jStart - 1] = cost;
      }
      // --- --- --- Stage 1: Up to the previous pruning point while advancing next_start: diag and top
      for (; j == next_start && j < prev_pp; ++j) {
        const auto d = dist(li, cols[j]);
        size_t dw = i > j ? i - j : j - i;  // abs(i-j)
        if (buffers[p + j - 1] <= buffers[p + j]) {
          // DIAG
          cost = buffers[p + j - 1] + d;
          mw = buffers_w[p + j - 1];
        } else {
          // TOP
          cost = buffers[p + j] + d;
          mw = std::max(dw, buffers_w[p + j]);
        }
        buffers[c + j] = cost;
        buffers_w[c + j] = mw;
        if (cost <= ub) { curr_pp = j + 1; } else { ++next_start; }
      }
      // --- --- --- Stage 2: Up to the previous pruning point without advancing next_start: left, diag and top
      for (; j < prev_pp; ++j) {
        const auto d = dist(li, cols[j]);
        size_t dw = i > j ? i - j : j - i;  // abs(i-j)
        if (cost <= buffers[p + j - 1] && cost <= buffers[p + j]) {
          // LEFT
          cost = cost + d;
          mw = std::max(dw, buffers_w[c + j - 1]);
        } else if (buffers[p + j - 1] <= cost && buffers[p + j - 1] <= buffers[p + j]) {
          // DIAG
          cost = buffers[p + j - 1] + d;
          mw = buffers_w[p + j - 1];
        } else {
          // TOP
          assert(buffers[p+j]<=cost && buffers[p+j]<=buffers[p+j-1]);
          cost = buffers[p + j] + d;
          mw = std::max(dw, buffers_w[p + j]);
        }
        buffers[c + j] = cost;
        buffers_w[c + j] = mw;
        if (cost <= ub) { curr_pp = j + 1; }
      }
      // --- --- --- Stage 3: At the previous pruning point. Check if we are within bounds.
      if (j < jStop) { // If so, two cases.
        const auto d = dist(li, cols[j]);
        size_t dw = i > j ? i - j : j - i;  // abs(i-j)
        if (j == next_start) { // Case 1: Advancing next start: only diag.
          // DIAG
          cost = buffers[p + j - 1] + d;
          mw = buffers_w[p + j - 1];
          buffers[c + j] = cost;
          buffers_w[c + j] = mw;
          if (cost <= ub) { curr_pp = j + 1; }
          else {
            // Special case if we are on the last alignment: return the actual cost if we are <= cutoff
            if (i == nblines - 1 && j == nbcols - 1 && cost <= cutoff) { return {cost, mw}; }
            else { return {POSITIVE_INFINITY, 0}; }
          }
        } else { // Case 2: Not advancing next start: possible path in previous cells: left and diag.
          if(cost <= buffers[p+j-1]){
            // LEFT
            cost = cost + d;
            mw = std::max(dw, buffers_w[c + j - 1]);
          } else {
            // DIAG
            cost = buffers[p + j - 1] + d;
            mw = buffers_w[p + j - 1];
          }
          buffers[c + j] = cost;
          buffers_w[c + j] = mw;
          if (cost <= ub) { curr_pp = j + 1; }
        }
        ++j;
      } else { // Previous pruning point is out of bound: exit if we extended next start up to here.
        if (j == next_start) {
          // But only if we are above the original UB
          // Else set the next starting point to the last valid column
          if (cost > cutoff) { return {POSITIVE_INFINITY, 0}; }
          else { next_start = nbcols - 1; }
        }
      }
      // --- --- --- Stage 4: After the previous pruning point: only prev.
      // Go on while we advance the curr_pp; if it did not advance, the rest of the line is guaranteed to be > ub.
      for (; j == curr_pp && j < jStop; ++j) {
        const auto d = dist(li, cols[j]);
        cost = cost + d;
        size_t dw = i > j ? i - j : j - i;  // abs(i-j)
        mw = std::max(dw, buffers_w[c + j - 1]); // Assess window when moving horizontally
        buffers[c + j] = cost;
        buffers_w[c + j] = mw;
        if (cost <= ub) { ++curr_pp; }
      }
      // --- --- ---
      prev_pp = curr_pp;
    } // End of main loop for(;i<nblines;++i)

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Finalisation
    // Check for last alignment (i==nblines implied, Stage 4 implies j<=nbcols). Cost must be <= original bound.
    if (j == nbcols && cost <= cutoff) { return {cost, mw}; }
    else { return {POSITIVE_INFINITY, 0}; }
  }

} // End of namespace tempo::univariate::fastee
