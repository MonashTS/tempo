#!/bin/python3
import pytempo.univariate.distances as ud
import numpy as np

s1 = np.array([2, 3, 4], dtype=float)
s2 = np.array([0, 0, 0, 0], dtype=float)

print(f"s1 of length {len(s1)} = {s1}\ns2 of length {len(s2)} = {s2}")

v = ud.dtw(s1, s2)
print(f"dtw(s1, s2) = {v}")

co1 = v+1
ea1 = ud.dtw(s1, s2, co1)
print(f"If cut-off > {v}, no early abandoning: dtw(s1, s2, {co1}) = {ea1}")

co2 = v
ea2 = ud.dtw(s1, s2, co2)
print(f"If cut-off = {v}, no early abandoning: dtw(s1, s2, {co2}) = {ea2}")

co3 = v-1
ea3 = ud.dtw(s1, s2, co3)
print(f"If cut-off < {v}, early abandoning: dtw(s1, s2, {co3}) = {ea3}")
