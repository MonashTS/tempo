#pragma once

#include "../utils.private.hpp"
#include "dtw_lb_keogh.hpp"

namespace tempo::distance {

  namespace univariate {

    /**  LB Webb - only applicable for same-length series.
     *   From the paper Webb GI, Petitjean F Tight lower bounds for Dynamic Time Warping.
     *   Based on https://github.com/GIWebb/DTWBounds
     *   NOTE: series sa and sb must have the same length (checked in debug mode)
     * @param sa                First series, as a pointer
     * @param length_sa         Length of the first series
     * @param upper_sa          Upper envelope of sa
     * @param lower_sa          Lower envelope of sa
     * @param lower_upper_sa    Lower envelope of upper_sa
     * @param upper_lower_sa    Upper envelope of lower_sa
     * @param sb                Second series, as a pointer
     * @param length_sb         Length of the second series
     * @param upper_sb          Upper envelope of sb
     * @param lower_sb          Lower envelope of sb
     * @param lower_upper_sb    Lower envelope of upper_sb
     * @param upper_lower_sb    Upper envelope of lower_sb
     * @param cfun              Cost Function utils::CFun<F>
     * @param w                 Warping window
     * @param cutoff                Upper bound (or "best so far")
     * @return The LB Webb value or +Inf if early abandoned
     */
    template<typename F>
    F lb_Webb(
      // Series A
      F const *sa, size_t length_sa,
      F const *upper_sa, F const *lower_sa,
      F const *lower_upper_sa, F const *upper_lower_sa,
      // Series B
      F const *sb, [[maybe_unused]] size_t length_sb,
      F const *upper_sb, F const *lower_sb,
      F const *lower_upper_sb, F const *upper_lower_sb,
      // Cost function
      utils::CFun<F> auto cfun,
      // Others
      size_t w, F cutoff
    ) {

      // --- --- ---
      assert(length_sa==length_sb);
      const auto length = length_sa;

      F lb = 0;
      size_t freeCountAbove = w;
      size_t freeCountBelow = w;

      for (size_t i{0}; i<length_sa&&lb<=cutoff; ++i) {
        // --- LB Keogh + counting free
        const auto sa_i = sa[i];
        const auto u_sb_i = upper_sb[i];
        if (sa_i>u_sb_i) {
          lb += cfun(sa_i, u_sb_i);
          freeCountBelow = u_sb_i>=upper_lower_sa[i] ? freeCountBelow + 1 : 0;
        } else {
          const auto l_sb_i = lower_sb[i];
          if (sa_i<l_sb_i) {
            lb += cfun(sa_i, l_sb_i);
            freeCountAbove = l_sb_i<=lower_upper_sa[i] ? freeCountAbove + 1 : 0;
          } else {
            freeCountAbove++;
            freeCountBelow++;
          }
        }

        // --- Add distance from sa to sb
        if (i>=w) {
          size_t j = i - w; // Unsigned always ok because i >= w
          const auto sb_j = sb[j];
          const auto u_sa_j = upper_sa[j];
          if (sb_j>u_sa_j) {
            if (freeCountAbove>w*2) {
              lb += cfun(sb_j, u_sa_j);
            } else {
              const auto ul_sb_j = upper_lower_sb[j];
              if (sb_j>ul_sb_j&&ul_sb_j>=u_sa_j) {
                lb += cfun(sb_j, u_sa_j) - cfun(ul_sb_j, u_sa_j);
              }
            }
          } else {
            const auto l_sa_j = lower_sa[j];
            if (sb_j<l_sa_j) {
              if (freeCountBelow>w*2) {
                lb += cfun(sb_j, l_sa_j);
              } else {
                const auto lu_sb_j = lower_upper_sb[j];
                if (sb_j<lu_sb_j&&lu_sb_j<=l_sa_j) {
                  lb += cfun(sb_j, l_sa_j) - cfun(lu_sb_j, l_sa_j);
                }
              }
            }
          }
        }
      }

      // --- --- ---
      // add distance from sb to sa for the last window's worth of positions
      for (size_t j = length - w; j<length&&lb<=cutoff; j++) {
        F sbj = sb[j];
        F u_sa_j = upper_sa[j];
        if (sbj>u_sa_j) {
          if (j>=length - freeCountAbove + w) {
            lb += cfun(sbj, u_sa_j);
          } else {
            F ul_sa_j = upper_lower_sb[j];
            if (sbj>ul_sa_j&&ul_sa_j>=u_sa_j) {
              lb += cfun(sbj, u_sa_j) - cfun(ul_sa_j, u_sa_j);
            }
          }
        } else {
          F l_sa_j = lower_sa[j];
          if (sbj<l_sa_j) {
            if (j>=length - freeCountBelow + w) {
              lb += cfun(sbj, l_sa_j);
            } else {
              F lu_sb_j = lower_upper_sb[j];
              if (sbj<lu_sb_j&&lu_sb_j<=l_sa_j) {
                lb += cfun(sbj, l_sa_j) - cfun(lu_sb_j, l_sa_j);
              }
            }
          }
        }
      }
      return (lb<=cutoff) ? lb : utils::PINF<F>;
    }


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Envelope computation
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    /** Given one series, creates all its envelope for lb Webb
     *  Uses the same envelope computation as lb Keogh */
    template<typename F>
    inline void get_keogh_envelopes_Webb(
      F const *series, size_t length, F *upper, F *lower, F *lower_upper, F *upper_lower, size_t w
    ) {
      get_keogh_envelopes(series, length, upper, lower, w);
      get_keogh_lo_envelope(upper, length, lower_upper, w);
      get_keogh_up_envelope(lower, length, upper_lower, w);
    }

    namespace {
      /** Create envelopes for lb Webb - vector version. Reallocation may occur! */
      template<typename F>
      void get_keogh_envelopes_Webb(std::vector<F> const& series,
                                    std::vector<F>& upper,
                                    std::vector<F>& lower,
                                    std::vector<F>& lower_upper,
                                    std::vector<F>& upper_lower,
                                    size_t w) {
        get_keogh_envelopes(series, upper, lower, w);
        get_keogh_lo_envelope(upper, lower_upper, w);
        get_keogh_up_envelope(lower, upper_lower, w);
      }
    }

  } // End of namespace univariate

} // End of namespace tempo::distance