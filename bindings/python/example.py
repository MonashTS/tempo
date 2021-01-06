#!/bin/python3

import pytempo
import numpy as np


def dir_mod(mod):
    print(f"dir({mod.__name__}) = {dir(mod)}")


dir_mod(pytempo)

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- TSeries
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
print("Create an empty series, convert to numpy array...")
ts0 = pytempo.TSeries()
array0 = np.array(ts0, copy=False)
print(array0)
print()

print("1D: Create a series based on a numpy array, and convert back")
array1D = np.array([1, 2, 3], dtype=float)
print(array1D)
ts1 = pytempo.TSeries(array1D, False, None)
print("Length = ", ts1.length())
array1D_bis = np.array(ts1, copy=False)
print(array1D_bis)
assert((array1D == array1D_bis).all())
print()

print("2D: Create a series based on a numpy array, and convert back")
array2D = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
print(array2D)
ts2 = pytempo.TSeries(array2D, False, None)
print("Length = ", ts2.length())
array2D_bis = np.array(ts2, copy=False)
print(array2D_bis)
assert((array2D == array2D_bis).all())
print()

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- Univariate
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
univariate = pytempo.univariate
dir_mod(univariate)

# --- --- --- Distance
ud = univariate.distances
dir_mod(ud)

s1 = np.array([2, 3, 4], dtype=float)
s2 = np.array([0, 0, 0, 0], dtype=float)

print(f"s1 of length {len(s1)} = {s1}\ns2 of length {len(s2)} = {s2}")

v = ud.dtw(s1, s2)
print(f"dtw(s1, s2) = {v}")

co1 = v + 1
ea1 = ud.dtw(s1, s2, co1)
print(f"If cut-off > {v}, no early abandoning: dtw(s1, s2, {co1}) = {ea1}")

co2 = v
ea2 = ud.dtw(s1, s2, co2)
print(f"If cut-off = {v}, no early abandoning: dtw(s1, s2, {co2}) = {ea2}")

co3 = v - 1
ea3 = ud.dtw(s1, s2, co3)
print(f"If cut-off < {v}, early abandoning: dtw(s1, s2, {co3}) = {ea3}")

# --- --- --- Transforms
ut = univariate.transforms
dir_mod(ut)

deriv_s1 = ut.derivative(s1)
deriv_s2 = np.empty(0)
ut.derivative(s2, deriv_s2)
print(f"Derivative of s1 = {deriv_s1}")
print(f"Derivative of s2 = {deriv_s2}")
