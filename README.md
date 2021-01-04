# tempo
Monash Time Series Classification Library

For now, this library only contains univariate C++ implementation of most common elastic distances
(the one used in Elastic Ensemble, Hive Cote, Proximity Forest and TSChief).
We also provide Python3 bindings.

## Elastic Distances

Our implementations have a O(n) space complexity and use pruning and early abandoning,
providing high performance and scalability.
They should provide a significant speed up in most of scenarios, even though there worst time complexity remain O(n²).
Note that they will not always be the fastest as early abandoning and pruning require some bookkeeping.
This bookkeeping must be compensated for, and this is not always possible (e.g. when the series are too short).

We have two versions of each distances: one with a cut-off point and one without.
The cut-off point is an upper bound above which pruning starts.
When "enough" pruning occurs (i.e. a full line of the cost matrix), the computation is early abandoned.
When no cut-off point is provided, we compute one ourself (based one the diagonal and the last column of the cost matrix)
before calling the version with cut-off.
This allows to prune computation even when no cut-off point is provided
(note that no early abandoning can occur under these conditions).


## Python3 Bindings
Bindings tested for Python3. All commands here assume a Python3 environment.
Ensure that the `wheel` packages is installed (e.g. with `pip install wheel`)
```bash
pip install wheel                             # To install wheel
python setup.py bdist_wheel                   # Build the package
pip install ./dist/pytempo-[info].whl         # Path to wheel
```

Distances must be called with numpy arrays.
Have a look at [bindings/python/example.py](bindings/python/example.py).

To remove the package:
```bash
pip uninstall pytempo.distances
```
