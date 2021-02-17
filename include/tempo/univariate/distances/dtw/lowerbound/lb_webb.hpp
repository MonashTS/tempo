#pragma once

#include "../../../../utils/utils.hpp"
#include "../../distances.hpp"
#include "envelopes.hpp"

namespace tempo::univariate {

    /** Create envelopes for lb Webb */
    [[maybe_unused]]static void get_keogh_envelopes_Webb(
            const double *series, size_t length,
            double *upper, double *lower,
            double *lower_upper, double *upper_lower,
            size_t w
    ) {
        get_keogh_envelopes(series, length, upper, lower, w);
        get_keogh_lo_envelope(upper, length, lower_upper, w);
        get_keogh_up_envelope(lower, length, upper_lower, w);
    }

    /** Create envelopes for lb Webb - vector version. Reallocation may occur! */
    inline void get_keogh_envelopes_Webb(
            const std::vector<double> &series,
            std::vector<double> &upper, std::vector<double> &lower,
            std::vector<double> &lower_upper, std::vector<double> &upper_lower,
            size_t w) {
        get_keogh_envelopes(series, upper, lower, w);
        get_keogh_lo_envelope(upper, lower_upper, w);
        get_keogh_up_envelope(lower, upper_lower, w);
    }

    /** LB Webb
     *   From the paper Webb GI, Petitjean F Tight lower bounds for Dynamic Time Warping.
     *   Based on https://github.com/GIWebb/DTWBounds
     *   NOTE: series sa and sb must have the same length (checked in debug mode)
     * @param sa First series, as a pointer
     * @param length_sa Length of the first series
     * @param upper_sa Upper envelope of sa
     * @param lower_sa Lower envelope of sa
     * @param lower_upper_sa Lower envelope of upper_sa
     * @param upper_lower_sa Upper envelope of lower_sa
     * @param sb Second series, as a pointer
     * @param length_sb Lenght of the second series
     * @param upper_sb Upper envelope of sb
     * @param lower_sb Lower envelope of sb
     * @param lower_upper_sb Lower envelope of upper_sb
     * @param upper_lower_sb Upper envelope of lower_sb
     * @param w Warping window
     * @param ub Upper bound (or "best so far")
     * @return The LB Webb value
     */
    template<auto dist = square_dist < double>>
    [[nodiscard]] inline double lb_Webb(
            // Series A
            const double *sa, size_t length_sa,
            const double *upper_sa, const double *lower_sa,
            const double *lower_upper_sa, const double *upper_lower_sa,
            // Series B
            const double *sb, [[maybe_unused]] size_t length_sb,
            const double *upper_sb, const double *lower_sb,
            const double *lower_upper_sb, const double *upper_lower_sb,
            // Others
            size_t w, double ub) {

        // --- --- ---
        assert(length_sa == length_sb);
        const auto length = length_sa;

        double lb = 0;
        size_t freeCountAbove = w;
        size_t freeCountBelow = w;

        for (size_t i{0}; i < length_sa && lb <= ub; ++i) {
            // --- LB Keogh + counting free
            const auto sa_i = sa[i];
            const auto u_sb_i = upper_sb[i];
            if (sa_i > u_sb_i) {
                lb += dist(sa_i, u_sb_i);
                freeCountBelow = u_sb_i >= upper_lower_sa[i] ? freeCountBelow + 1 : 0;
            } else {
                const auto l_sb_i = lower_sb[i];
                if (sa_i < l_sb_i) {
                    lb += dist(sa_i, l_sb_i);
                    freeCountAbove = l_sb_i <= lower_upper_sa[i] ? freeCountAbove + 1 : 0;
                } else {
                    freeCountAbove++;
                    freeCountBelow++;
                }
            }

            // --- Add distance from sa to sb
            if (i >= w) {
                size_t j = i - w; // Unsigned always ok because i >= w
                const auto sb_j = sb[j];
                const auto u_sa_j = upper_sa[j];
                if (sb_j > u_sa_j) {
                    if (freeCountAbove > w * 2) {
                        lb += dist(sb_j, u_sa_j);
                    } else {
                        const auto ul_sb_j = upper_lower_sb[j];
                        if (sb_j > ul_sb_j && ul_sb_j >= u_sa_j) {
                            lb += dist(sb_j, u_sa_j) - dist(ul_sb_j, u_sa_j);
                        }
                    }
                } else {
                    const auto l_sa_j = lower_sa[j];
                    if (sb_j < l_sa_j) {
                        if (freeCountBelow > w * 2) {
                            lb += dist(sb_j, l_sa_j);
                        } else {
                            const auto lu_sb_j = lower_upper_sb[j];
                            if (sb_j < lu_sb_j && lu_sb_j <= l_sa_j) {
                                lb += dist(sb_j, l_sa_j) - dist(lu_sb_j, l_sa_j);
                            }
                        }
                    }
                }
            }
        }

        // --- --- ---
        // add distance from sb to sa for the last window's worth of positions
        for (size_t j = length - w; j < length && lb <= ub; j++) {
            double sbj = sb[j];
            double u_sa_j = upper_sa[j];
            if (sbj > u_sa_j) {
                if (j >= length - freeCountAbove + w) {
                    lb += dist(sbj, u_sa_j);
                } else {
                    double ul_sa_j = upper_lower_sb[j];
                    if (sbj > ul_sa_j && ul_sa_j >= u_sa_j) {
                        lb += dist(sbj, u_sa_j) - dist(ul_sa_j, u_sa_j);
                    }
                }
            } else {
                double l_sa_j = lower_sa[j];
                if (sbj < l_sa_j) {
                    if (j >= length - freeCountBelow + w) {
                        lb += dist(sbj, l_sa_j);
                    } else {
                        double lu_sb_j = lower_upper_sb[j];
                        if (sbj < lu_sb_j && lu_sb_j <= l_sa_j) {
                            lb += dist(sbj, l_sa_j) - dist(lu_sb_j, l_sa_j);
                        }
                    }
                }
            }
        }
        return lb;
    }


    template<auto dist = square_dist < double>>
    [[nodiscard]] inline double lb_Webb(
            // Series A
            const std::vector<double> &sa,
            const std::vector<double> &upper_sa, const std::vector<double> &lower_sa,
            const std::vector<double> &lower_upper_sa, const std::vector<double> &upper_lower_sa,
            // Series B
            const std::vector<double> &sb,
            const std::vector<double> &upper_sb, const std::vector<double> &lower_sb,
            const std::vector<double> &lower_upper_sb, const std::vector<double> &upper_lower_sb,
            // Others
            size_t w, double ub) {
        return lb_Webb < dist > (sa.data(), sa.size(),
                                 upper_sa.data(), lower_sa.data(), lower_upper_sa.data(), upper_lower_sa.data(),
                                 sb.data(), sb.size(),
                                 upper_sb.data(), lower_sb.data(), lower_upper_sb.data(), upper_lower_sb.data(),
                                 w, ub
        );
    }


} // End of namespace tempo::univariate