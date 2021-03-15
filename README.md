# Tempo
Monash Time Series Classification Library

For now, this library only contains univariate C++ implementation of most common elastic distances
used in Elastic Ensemble, Hive Cote, Proximity Forest and TSChief.
 * DTW, CDTW and WDTW
 * ERP
 * LCSS
 * MSM
 * Squared Euclidean Distance
 * TWE
 
The derivative version of DTW, CDTW and WDTW are not directly implemented.
We rather use the concept of "transforms", which amount to a pre-processing step.
Currently available transforms are:
  * Upper and lower envelope computation (to use with DTW and CDTW lower bounds)
  * Derivative
  
Hence, you can easily setup a derivative version of any distance.
See "[Demo application](#demo-application)" below.
 
We also provide [Python3 bindings](#python3-bindings).

## Note
Tempo is young and in active development. Do not hesitate to repport issues!

## Elastic Distances

Our implementations have a O(n) space complexity and tightly integrate pruning with early abandoning, providing high performance and scalability.
They should provide a significant speed up in most of scenarios, even though there worst time complexity remain O(n²).
Note that they will not always be the fastest as early abandoning and pruning require some bookkeeping.
This bookkeeping must be compensated for, and this is not always possible (e.g. when the series are too short, with windows too small).

We have two versions of each distances: one with a cut-off point and one without.
The cut-off point is an upper bound above which pruning starts.
When "enough" pruning occurs (i.e. a full line of the cost matrix), the computation is early abandoned.
When no cut-off point is provided, we compute one ourself (based one the diagonal and the last column of the cost matrix)
before calling the version with cut-off.
This allows to prune computation even when no cut-off point is provided
(note that no early abandoning can occur under these conditions).

## Demo application

The folder [app/nn1dist](app/nn1dist) contains a demo application for nn1 classification.
To build it under linux, do

```bash
mkdir cmake-build-release
cd cmake-build-release
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make
./nn1dist
```

The last line will print a help message.

This application is designed to work with the TS format from [timeseriesclassification.com](http://timeseriesclassification.com/dataset.php).
You can either give the path to a UCR Archive folder and dataset name (flag `-ucr`), or directly the train and test files (flag `-tt`).
The distance to use is specified with the ``-dist`` flag, followed by the distance arguments (see the help message).

A nth derivative can be specified with the ``-tr derivative n`` flag.
So
```bash
./nn1dist -dist msm 0.05 -ucr ~/Univariate_ts/ Crop -tr derivative 1
```
will compute a "Derivative MSM".


## Usage with CMake and C++
Download the sources and link to Tempo adding in your CMakeLists.txt:
```cmake
add_subdirectory(path_to_tempo)
target_link_libraries(<target> PRIVATE tempo)
```

## Python3 Bindings
Bindings tested for Python3. All commands here assume a Python3 environment.
Ensure that the `wheel` packages is installed (e.g. with `pip install wheel`)
```bash
pip install wheel                             # To install wheel
python setup.py bdist_wheel                   # Build the package
pip install ./dist/pytempo-[info].whl         # Path to wheel ( --force-reinstall to overwrite a previous installation)
```

Distances must be called with numpy arrays.
Have a look at [bindings/python/example.py](bindings/python/example.py).

To remove the package:
```bash
pip uninstall pytempo.distances
```

Python bindings are done thanks to [pybind11](https://github.com/pybind/pybind11)

## Benckmarks
The algorithm is submitted in a paper under review.
The paper contains benchmarks that can be found here https://github.com/HerrmannM/paper-2021-EAPElasticDist.

## References
See [REFERENCE.md](REFERENCE.md)

