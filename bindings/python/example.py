#!/bin/python3

import pytempo
import numpy as np
from math import nan, inf

# Flags to (de)activate section of the example
doPrintMod = False
doTSeries = False
doUniDist = True
doUniTransform = False


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Print module
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def dir_mod(mod):
    print(f"dir({mod.__name__}) = {dir(mod)}")

def funPrintMod():
    dir_mod(pytempo)
    dir_mod(pytempo.univariate)
    dir_mod(pytempo.univariate.distances)
    dir_mod(pytempo.univariate.transforms)

if doPrintMod:
    funPrintMod()



# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- TSeries
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def funTSeries():
    print()
    print("TSeries: Create an empty series, convert to a numpy array. Use copy=False for direct access.")
    ts0 = pytempo.TSeries()
    array0 = np.array(ts0, copy=False)
    print(array0)
    print()

    print()
    print("TSeries 1D (univariate): Create a series based on a numpy array, and convert back")
    print("Note: we tell pyTempo that there are no missing data")
    array1D_origin = np.array([1, 2, 3], dtype=float)
    array1D = array1D_origin.copy()
    print(array1D)
    ts1 = pytempo.TSeries(array1D, False, "my label")
    print()

    print("Access the TSeries")
    print("Length = ", ts1.length())
    print("Dim = ", ts1.ndim())
    print("Has missing = ", ts1.has_missing())
    print("Label = ", ts1.label())
    print("Access [0,0] = ", ts1[0, 0])
    print("Univariate access [0] = ", ts1[0])
    print()

    print("Resizing the array must fail (refcheck=True by default)")
    print("Do not resize with refcheck = False! It will invalidate the memory reference on the C++ side")
    try:
        array1D.resize((100000,), refcheck=True)
        print(array1D)
    except ValueError as err:
        print("ValueError Exception:")
        print(err)
    print()

    print("Getting an array back from a TSeries")
    print("'copy=False' prevents copying, but also prevents us from modifying the series through the resulting array")
    print("TSeries uses the 'Buffer Protocol', and this applied to anything obtained this way")
    array1D_back = np.array(ts1, copy=False)
    try:
        array1D_back[0] = 7
        print("array: ", array1D, " from TSeries, modified: ", array1D_back)
    except ValueError as err:
        print("ValueError Exception:")
        print(err)
    print("Use 'copy=True' if you want to modify the data")
    array1D_back_copy = np.array(ts1, copy=True)
    array1D_back_copy[0] = 7
    print("array: ", array1D, " from TSeries, modified: ", array1D_back_copy)
    print()

    print("Reshaping the array will not impact the TSeries (same data)")
    try:
        array1D = array1D.reshape((1, 3))
        print(array1D.shape)
    except ValueError as err:
        print("ValueError Exception:")
        print(err)
    print()

    print("Changing stored values will be reflected in the TSeries")
    print("Note that we changed the shape of array1D: broadcasting rewrites everything behind index 0!")
    array1D[0] = 5
    print("array: ", array1D, " from TSeries: ", array1D_back)
    print("Another update")
    array1D[0][1] = 10
    print("array: ", array1D, " from TSeries: ", array1D_back)
    try:
        assert (array1D_origin == array1D_back).all()
    except AssertionError as err:
        print("Assertion error (origin != back)")
        print(err)
    print()

    print()
    print("TSeries 2D (multivariate): Create a series based on a numpy array, and convert back")
    print("Note: we let pyTempo chek if there are missing data or not")
    array2D = np.array([[1, 2, 3], [4, 5, nan]], dtype=float)
    print(array2D)
    ts2 = pytempo.TSeries(array2D, None)

    print("Access the TSeries")
    print("Length = ", ts2.length())
    print("Dim = ", ts2.ndim())
    print("Has missing = ", ts2.has_missing())
    print("Label = ", ts2.label())
    print("Access [1,2] = ", ts2[1, 2])
    print("Univariate access [0] = ...")
    try:
        print(ts2[0])
    except ValueError as err:
        print("ValueError Exception:")
        print(err)
    print()

    print("Getting an array back from a TSeries")
    print("Note: nan!=nan, adapt the condition")
    array2D_back = np.array(ts2, copy=False)
    print("array:")
    print(array2D)
    print("from TSeries:")
    print(array2D_back)
    assert ((array2D == array2D_back) | (np.isnan(array2D) & np.isnan(array2D_back))).all()
    print()
    # --- End of funTSeries()

if doTSeries:
    funTSeries()

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- Univariate Distance & Lower Bounds
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def funUniDist():
    print()

    # --- --- --- Distance
    ud = pytempo.univariate.distances

    # You can also use lists!
    query = [4,7,5,30,1,9,7,5]
    candidate = [3,15,6,9,7,2,6,5]

    print(f"query of length {len(query)} = {query}")
    print(f"candidate of length {len(candidate)} = {candidate}")

    print()
    print("Demonstrating cut-off with DTW:")
    v = ud.dtw(query, candidate)
    print(f"  DTW(query, candidate) = {v}")

    co1 = v + 1
    ea1 = ud.dtw(query, candidate, co1)
    print(f"    If cut-off > {v}, no early abandoning: DTW(query, candidate, {co1}) = {ea1}")

    co2 = v
    ea2 = ud.dtw(query, candidate, co2)
    print(f"    If cut-off = {v}, no early abandoning: DTW(query, candidate, {co2}) = {ea2}")

    co3 = v - 1
    ea3 = ud.dtw(query, candidate, co3)
    print(f"    If cut-off < {v},    early abandoning: DTW(query, candidate, {co3}) = {ea3}")

    print()
    w = 2
    print("Other distances:")
    print(f"  CDTW w={w}: ", ud.cdtw(query, candidate, w))
    print(f"  ERP gValue = 0.1, w = {w}: ", ud.erp(query, candidate, 0.1, w))
    print(f"  LCSS epsilon = 0.01, w = {w}: ", ud.lcss(query, candidate, 0.01, w))
    print(f"  MSM cost = 0.5: ", ud.msm(query, candidate, 0.5))
    print(f"  SQED: ", ud.squaredED(query, candidate))
    print(f"  TWE nu = 0.1, lambda = 0.2 : ", ud.twe(query, candidate, 0.1, 0.2))

    # WDTW needs the weights
    weights = ud.wdtw_weights(max(len(query), len(candidate)), 0.5)
    print("  WDTW: generating weights with g=0.5: ", weights)
    print("  WDTW with generated weights: ", ud.wdtw(query, candidate, weights))

    # Envelopes computation & lower bound
    print()
    print(f"DTW/CDTW envelopes & lower bounds (w={w})")


    (q_up, q_lo) = ud.get_envelopes(query, w)
    (c_up, c_lo) = ud.get_envelopes(candidate, w)
    print("query = ", query)
    print(f"Query up = ", q_up, f" Query lo = ", q_lo)
    print("Candidate = ", candidate)
    print(f"Candidate up = ", c_up, f" Candidate lo = ", c_lo)

    print()
    print("LB-Keogh query, envelope from candidate, no cutoff: ", ud.lb_keogh(query, c_up, c_lo))
    print("LB-Keogh query, envelope from candidate, cutoff = 0: ", ud.lb_keogh(query, c_up, c_lo, 0))
    print("'inf' indicates that the computation of the lower bound has been early abandoned")
    print("The computation is only abandoned if result > cutoff")
    print("LB-Keogh query, envelope from candidate, cutoff = 1: ", ud.lb_keogh(query, c_up, c_lo, 1))

    print()
    print("LB-Keogh candidate, envelope from query, no cutoff: ", ud.lb_keogh(candidate, q_up, q_lo))

    print()
    print("LB-Keogh2: cascading LB-Keogh(query, envs from candidate) with LB-Keogh(candidate, envs from query)")
    print("not cutoff: ", ud.lb_keogh2(query, q_up, q_lo, candidate, c_up, c_lo))
    print("cutoff = 5: ", ud.lb_keogh2(query, q_up, q_lo, candidate, c_up, c_lo, 5))
    print("No cutoff actually means cutoff = +Infinity.")
    print("The cascading only takes place if the first computation is below (or equal to) the cutoff.")
    print("In the first case, we have 226 <= +Infinity: the computation is cascaded, and we then find 1")
    print("After cascading, the max value is returned (226)")
    print("In the second case, we have 226 > 5: the computation is early abandoned")

    print()
    print("LB-Keogh2j: 'Joined' LB-Keogh2. Instead of cascading two LB-Keoghs, do them at the same time!")
    print("not cutoff: ", ud.lb_keogh2j(query, q_up, q_lo, candidate, c_up, c_lo))
    print("cutoff = 5: ", ud.lb_keogh2j(query, q_up, q_lo, candidate, c_up, c_lo, 5))

    print()
    print("LB-Enhanced: compute 'v' bands (5 by default) on each side of the cost matrix, and 'bridge' with LB-Keogh")
    print("not cutoff: ", ud.lb_enhanced(query, candidate, c_up, c_lo, w))
    print("cutoff = 5: ", ud.lb_enhanced(query, candidate, c_up, c_lo, w, 5))
    print("In this case, our series are too short for the number of bands.")
    print("The number is adjusted to v=4 on each side, and only the band are computed (no LB-Keogh bridging)")
    print("Now, with v = 3")
    print("not cutoff: ", ud.lb_enhanced(query, candidate, c_up, c_lo, w, v=3))
    print("cutoff = 5: ", ud.lb_enhanced(query, candidate, c_up, c_lo, w, 5, 1))
    print("In the first case, the tightness is much better! Note that v=5 is usually a safe bet (see the paper)")
    print("v=2, not cutoff: ", ud.lb_enhanced(query, candidate, c_up, c_lo, w, v=2))
    print("v=1, not cutoff: ", ud.lb_enhanced(query, candidate, c_up, c_lo, w, v=1))
    print("v=0, not cutoff: ", ud.lb_enhanced(query, candidate, c_up, c_lo, w, v=0))

    print()
    print("LB-Enhanced2j: Just like LB-Enhanced, but with a 'joined' LB-Keogh2j bridge instead of a simple LB-Keogh")
    print("not cutoff: ", ud.lb_enhanced2j(query, q_up, q_lo, candidate, c_up, c_lo, w))
    print("cutoff = 5: ", ud.lb_enhanced2j(query, q_up, q_lo, candidate, c_up, c_lo, w, 5))
    print("In this case, our series are too short for the number of bands.")
    print("The number is adjusted to v=4 on each side, and only the band are computed (no LB-Keogh bridging)")
    print("Now, with v = 3")
    print("not cutoff: ", ud.lb_enhanced2j(query, q_up, q_lo, candidate, c_up, c_lo, w, v=3))
    print("cutoff = 5: ", ud.lb_enhanced2j(query, q_up, q_lo, candidate, c_up, c_lo, w, 5, 1))
    print("v=2, not cutoff: ", ud.lb_enhanced2j(query, q_up, q_lo, candidate, c_up, c_lo, w, v=2))
    print("v=1, not cutoff: ", ud.lb_enhanced2j(query, q_up, q_lo, candidate, c_up, c_lo, w, v=1))
    print("v=0, not cutoff: ", ud.lb_enhanced2j(query, q_up, q_lo, candidate, c_up, c_lo, w, v=0))
    print("Here, there is no difference with LB-Enhanced, but LB-Enhanced2j is usually better.")

    print()
    print("LB-Webb: requires more envelopes: upper of lower and lower of upper")
    q_lo_up = ud.get_envelope_lower(q_up, w) # Lower of upper
    q_up_lo = ud.get_envelope_upper(q_lo, w) # Upper of lower
    c_lo_up = ud.get_envelope_lower(c_up, w)
    c_up_lo = ud.get_envelope_upper(c_lo, w)
    print("no cutoff: ", ud.lb_webb(query, q_up, q_lo, q_lo_up, q_up_lo, candidate, c_up, c_lo, c_lo_up, c_up_lo, w))
    print("cutoff = 5: ", ud.lb_webb(query, q_up, q_lo, q_lo_up, q_up_lo, candidate, c_up, c_lo, c_lo_up, c_up_lo, w, 5))
    # --- end of funUniDist()

if doUniDist:
    funUniDist()

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- Univariate Transforms
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def funUniTransform():
    print()

    ut = pytempo.univariate.transforms

    s1 = np.array([2, 3, 4], dtype=float)
    s2 = np.array([0, 0, 0, 0], dtype=float)

    print(f"s1 of length {len(s1)} = {s1}")
    print(f"s2 of length {len(s2)} = {s2}")

    deriv_s1 = ut.derivative(s1)
    deriv_s2 = np.empty(0)
    ut.derivative(s2, deriv_s2)
    print(f"Derivative of s1 = {deriv_s1}")
    print(f"Derivative of s2 = {deriv_s2}")
    # --- end of funUniTransform()

if doUniTransform:
    funUniTransform()