import numpy as np
import math
import pytest
from pytempo.univariate import distance as td
from pytempo.univariate import transform as tt

s1 = np.array(
    [0.24, 0.257, 0.274, 0.257, 0.277, 0.297, 0.317, 0.29325, 0.2695, 0.24575, 0.222, 0.19, 0.229, 0.246, 0.28, 0.334,
     0.388, 0.458, 0.528, 0.615, 0.717, 0.727, 0.753, 0.725, 0.743, 0.761, 0.77233, 0.78367, 0.795, 0.771, 0.749, 0.705,
     0.702, 0.642, 0.5415, 0.441, 0.41475, 0.3885, 0.36225, 0.336, 0.324, 0.312, 0.331, 0.35, 0.333, 0.316])

s2 = np.array(
    [0.219, 0.209, 0.199, 0.21567, 0.23233, 0.249, 0.237, 0.185, 0.203, 0.221, 0.239, 0.251, 0.423, 0.463, 0.55, 0.6535,
     0.757, 0.79, 0.83, 0.828, 0.818, 0.802, 0.815, 0.809, 0.803, 0.797, 0.80033, 0.80367, 0.807, 0.792, 0.749, 0.713,
     0.677, 0.612, 0.494, 0.376, 0.369, 0.362, 0.355, 0.348, 0.405, 0.351, 0.295, 0.239, 0.218, 0.197])

cfe = 2.0
w = int(max(len(s2), len(s1)) / 10) + 1
nb_bands = 5
assert(nb_bands <= len(s2))
adtw_p = 0.025
wdtw_g0 = 0.01
wdtw_g1 = 0.1
erp_gv = 0.01
lcss_e = 0.01
msm_c0 = 0.05
msm_c1 = 0.1
twe_nu0 = 0.01
twe_nu1 = 0.05
twe_lambda0 = 0.001
twe_lambda1 = 0.005

print()
print(dir(td))
print(dir(tt))


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# DTW & direct alignment
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

directa = td.directa(s1, s2, cfe)
print(f"direct alignment = {directa}")

dtw_r0 = td.dtw(s1, s2, cfe, 0)
print(f"dtw r0, no window, no cutoff = {dtw_r0}")

dtw_r1 = td.dtw(s1, s2, cfe)
print(f"dtw r1, no window, no cutoff = {dtw_r1}")

dtw_r2 = td.dtw(s1, s2, cfe, w)
print(f"dtw r2, window {w}, no cutoff = {dtw_r2}")


def test_dtw():
    assert dtw_r0 == directa, "Window of 0 must be equal to direct alignment"
    assert dtw_r0 >= dtw_r1, "Smaller window should result in larger distance"
    assert td.dtw(s1, s2, cfe, None) == dtw_r1, "Test optional window argument"
    assert dtw_r1 != math.inf, "Should not early abandoned"
    assert td.dtw(s1, s2, cfe, None, math.inf) == dtw_r1, "Should have same result - cutoff = +inf"

(env_upper_s1, env_lower_s1) = td.envelopes(s1, w)
(env_upper_s2, env_lower_s2) = td.envelopes(s2, w)

def test_envelopes():
    assert np.array_equal(env_upper_s1, td.upper_envelope(s1, w))
    assert np.array_equal(env_lower_s1, td.lower_envelope(s1, w))

lbkeogh1 = td.lb_keogh(s1, env_upper_s2, env_lower_s2, cfe)
print(f"LB Keogh(s1, upper(s2), lower(s2) = {lbkeogh1}")

lbkeogh2 = td.lb_keogh(s2, env_upper_s1, env_lower_s1, cfe)
print(f"LB Keogh(s2, upper(s1), lower(s1) = {lbkeogh2}")

lbkeogh3 = td.lb_keogh2j(s1, env_upper_s1, env_lower_s1, s2, env_upper_s2, env_lower_s2, cfe)
print(f"LB Keogh2j(s1 + envelopes, s2 + envelopes) = {lbkeogh2}")


def test_lb_keogh():
    assert lbkeogh1 <= dtw_r2
    assert lbkeogh2 <= dtw_r2
    assert lbkeogh3 <= dtw_r2
    assert lbkeogh3 >= lbkeogh1, "LBKeogh2j must be as tight or tighter than LBKeogh"
    assert lbkeogh3 >= lbkeogh2, "LBKeogh2j must be as tight or tighter than LBKeogh"

lbenhanced1 = td.lb_enhanced(s1, s2, env_upper_s2, env_lower_s2, cfe, 0, w)
print(f"LB Enhanced(s1, s2 and envelopes, nb bands={0}) = {lbenhanced1}")

lbenhanced2 = td.lb_enhanced(s1, s2, env_upper_s2, env_lower_s2, cfe, nb_bands, w)
print(f"LB Enhanced(s1, s2 and envelopes, nb bands = {nb_bands}) = {lbenhanced2}")

lbenhanced3 = td.lb_enhanced2j(s1, env_upper_s1, env_lower_s1, s2, env_upper_s2, env_lower_s2, cfe, nb_bands, w)
print(f"LB Enhanced2j(s1 + envelopes, s2  + envelopes, nb bands = {nb_bands}) = {lbenhanced3}")

lbenhanced4 = td.lb_enhanced2j(s1, env_upper_s1, env_lower_s1, s2, env_upper_s2, env_lower_s2, cfe, int(len(s1)/2), w)
print(f"LB Enhanced2j(s1 + envelopes, s2  + envelopes, nb bands = max band) = {lbenhanced4}")


def test_lb_enhanced():
    assert lbenhanced1 >= lbkeogh1, "LBEnhanced with 0 bands must be tighter than same as LB Keogh (with same series/envelopes)"
    assert lbenhanced2 >= lbenhanced1, "LBEnhanced must be tighter with more bands"
    assert lbenhanced3 >= lbenhanced2, "LBEnhanced2j must be tighter than lb LBEnhanced"
    assert lbenhanced3 >= lbkeogh3, "LBEnhanced2j must be tighter than LBKeogh2j"
    assert lbenhanced3 <= dtw_r2
    with pytest.raises(Exception): # Must raise an exception: too many bands
        td.lb_enhanced2j(s1, env_upper_s1, env_lower_s1, s2, env_upper_s2, env_lower_s2, cfe, int(len(s1)/2)+1, w)


lbenhanced4 = td.lb_enhanced2j(s1, env_upper_s1, env_lower_s1, s2, env_upper_s2, env_lower_s2, cfe, int(len(s1)/2), w)
print(f"LB Enhanced2j(s1 + envelopes, s2  + envelopes, nb bands = max band) = {lbenhanced4}")

env_webb_1 = td.webb_envelopes(s1, w)
(u1, l1, lu1, ul1) = env_webb_1
env_webb_2 = td.webb_envelopes(s2, w)
(u2, l2, lu2, ul2) = env_webb_2

# Example of how the produced tuples can be used in arguments
webb1 = td.lb_webb(s1, *env_webb_1, s2, *env_webb_2, cfe, w)
print(f"LB Webb(s1 + envelopes, s2 + envelopes) = {webb1}")

def test_webb_envelopes():
        assert np.array_equal(u1, env_upper_s1)
        assert np.array_equal(l1, env_lower_s1)
        assert np.array_equal(u2, env_upper_s2)
        assert np.array_equal(l2, env_lower_s2)

def test_webb():
    assert webb1 >= lbkeogh3, "LB Webb is tighter than LB Keogh"
    assert webb1 <= dtw_r2

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# ADTW
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

adtw_r0 = td.adtw(s1, s2, cfe, 0)
print(f"adtw r0, penalty 0, no cutoff = {adtw_r0}")
adtw_r1 = td.adtw(s1, s2, cfe, math.inf)
print(f"adtw r1, penalty +inf, no cutoff = {adtw_r1}")
adtw_r2 = td.adtw(s1, s2, cfe, adtw_p)
print(f"adtw r2, penalty {adtw_p}, no cutoff = {adtw_r2}")


def test_adtw():
    assert adtw_r0 == dtw_r1, "Penalty of 0 should equal DTW NO WINDOW"
    assert adtw_r1 == dtw_r0, "Penalty of +inf should equal DTW with WINDOW=0"
    assert td.adtw(s1, s2, cfe, adtw_p, adtw_r2 - 0.05) == math.inf, "Should early abandoned - cutoff < adtw_r2"
    assert td.adtw(s1, s2, cfe, adtw_p, adtw_r2) == adtw_r2, "Should have same result - cutoff = adtw_r2"
    assert td.adtw(s1, s2, cfe, adtw_p, adtw_r2 + 0.05) == adtw_r2, "Should have same result - cutoff > adtw_r2"

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# WDTW
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

weights_0 = td.wdtw_weights(wdtw_g0, len(s1))
weights_1 = td.wdtw_weights(wdtw_g1, len(s1))

def test_wdtw_weights():
    assert weights_0[0] < weights_0[-1]
    assert weights_1[0] < weights_1[-1]
    assert weights_0[-1] - weights_0[0] < weights_1[-1] - weights_1[0], "weight difference must be larger for larger g"
    ww = td.wdtw_weights(0, len(s1))
    assert np.all(ww == ww[0]), "g=0 should give constant weights"


wdtw_r0 = td.wdtw(s1, s2, cfe, weights_0)
print(f"wdtw r0, weights {wdtw_g0}, no cutoff = {wdtw_r0}")

wdtw_r1 = td.wdtw(s1, s2, cfe, weights_1)
print(f"wdtw r1, weights {wdtw_g1}, no cutoff = {wdtw_r1}")

def test_wdtw():
    assert wdtw_r0 >= wdtw_r1, "Larger g leads to smaller results"



# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# ERP
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

erp_r0 = td.erp(s1, s2, cfe, erp_gv)
print(f"erp r0, gv {erp_gv}, no window, no cutoff = {erp_r0}")
erp_r1 = td.erp(s1, s2, cfe, erp_gv, w)
print(f"erp r1, gv {erp_gv}, window {w}, no cutoff = {erp_r1}")


def test_erp():
    assert td.erp(s1, s2, cfe, erp_gv, 0) == dtw_r0, "Should be the same as DTW for w=0"


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# LCSS
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

lcss_r0 = td.lcss(s1, s2, lcss_e)
print(f"lcss r0, epsilon {lcss_e}, no window, no cutoff = {lcss_r0}")
lcss_r1 = td.lcss(s1, s2, lcss_e, w)
print(f"lcss r1, epsilon {lcss_e}, window {w}, no cutoff = {lcss_r1}")


def test_lcss():
    assert 0 <= lcss_r0 <= 1, "LCSS result should be between 0 and 1"
    assert lcss_r1 >= lcss_r0, "LCSS with window should be more constrained (greater distance)"

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# MSM
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

msm_r0 = td.msm(s1, s2, msm_c0)
print(f"msm r0, cost {msm_c0}, no cutoff = {msm_r0}")
msm_r1 = td.msm(s1, s2, msm_c1)
print(f"msm r1, cost {msm_c1}, no cutoff = {msm_r1}")


def test_msm():
    assert msm_r1 >= msm_r0, "MSM with higher cost should be more constrained (greater distance)"

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# TWE
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

twe_r00 = td.twe(s1, s2, twe_nu0, twe_lambda0)
print(f"twe r00, nu {twe_nu0}, lambda {twe_lambda0}, no cutoff = {twe_r00}")

twe_r01 = td.twe(s1, s2, twe_nu0, twe_lambda1)
print(f"twe r01, nu {twe_nu0}, lambda {twe_lambda1}, no cutoff = {twe_r01}")

twe_r10 = td.twe(s1, s2, twe_nu1, twe_lambda0)
print(f"twe r10, nu {twe_nu1}, lambda {twe_lambda0}, no cutoff = {twe_r10}")

twe_r11 = td.twe(s1, s2, twe_nu1, twe_lambda1)
print(f"twe r11, nu {twe_nu1}, lambda {twe_lambda1}, no cutoff = {twe_r11}")

def test_twe():
    assert twe_r01 >= twe_r00, "TWE with higher penalty should be more constrained (greater distance)"
    assert twe_r10 >= twe_r00, "TWE with higher stiffness should be more constrained (greater distance)"
    assert twe_r11 >= twe_r01, "TWE with higher penalty should be more constrained (greater distance)"
    assert twe_r11 >= twe_r10, "TWE with higher stiffness should be more constrained (greater distance)"

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Derivative
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

d1 = tt.derive(s1)
print(d1)
