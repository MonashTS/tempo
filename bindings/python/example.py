#!/bin/python3

import pytempo
import numpy as np
from math import nan

# Flags to (de)activate section of the example
doPrintMod = True
doTSeries = True
doUniDist = True
doUniTransform = True


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
# --- --- --- Univariate
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def funUniDist():
    print()

    # --- --- --- Distance
    ud = pytempo.univariate.distances

    s1 = np.array([2, 3, 4], dtype=float)
    s2 = np.array([0, 0, 0, 0], dtype=float)

    print(f"s1 of length {len(s1)} = {s1}")
    print(f"s2 of length {len(s2)} = {s2}")

    print()
    print("Demonstrating cut-off with DTW:")
    v = ud.dtw(s1, s2)
    print(f"  DTW(s1, s2) = {v}")

    co1 = v + 1
    ea1 = ud.dtw(s1, s2, co1)
    print(f"    If cut-off > {v}, no early abandoning: DTW(s1, s2, {co1}) = {ea1}")

    co2 = v
    ea2 = ud.dtw(s1, s2, co2)
    print(f"    If cut-off = {v}, no early abandoning: DTW(s1, s2, {co2}) = {ea2}")

    co3 = v - 1
    ea3 = ud.dtw(s1, s2, co3)
    print(f"    If cut-off < {v},    early abandoning: DTW(s1, s2, {co3}) = {ea3}")

    print()
    print("Other distances:")
    print("  CDTW w=3: ", ud.cdtw(s1, s2, 3))
    print("  ERP gValue = 0.1, w = 3: ", ud.erp(s1, s2, 0.1, 3))
    print("  LCSS epsilon = 0.01, w = 3: ", ud.lcss(s1, s2, 0.01, 1))
    print("  MSM cost = 0.5: ", ud.msm(s1, s2, 0.5))
    print("  SQED: ", ud.squaredED(s1, s2))
    print("  TWE nu = 0.1, lambda = 0.2 : ", ud.twe(s1, s2, 0.1, 0.2))

    # WDTW needs the weights
    weights = ud.wdtw_weights(max(len(s1), len(s2)), 0.5)
    print("  WDTW: generating weights with g=0.5: ", weights)
    print("  WDTW with generated weights: ", ud.wdtw(s1, s2, weights))
    # --- end of funUniDist()

if doUniDist:
    funUniDist()

# --- --- --- Transforms
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