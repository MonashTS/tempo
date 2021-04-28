# Tempo ![logo](./doc/logo/tempo100.png)
Monash Time Series Classification Library.
Monash University, Melbourne, Australia.

Tempo is not only a project fo researchers!
Our aim is to deliver a user-friendly library,
allowing anyone to enjoy our last work in the domain of time series classification.

**Tempo is young and in active development. Do not hesitate to report issues!**

# Features

## Elastic distances (C++, Java, Python3)
Starting point of our library, we provide implementations of the most commonly used elastic distances.
Our implementations in C++ and Java are early abandoned and pruned, and we offer [Python3 bindings](#python3-bindings).
Available distances are
 * DTW and CDTW, with lower bounds (Lb Keogh, Lb Enhanced, Lb Webb)
 * WDTW
 * ERP
 * LCSS
 * MSM
 * Squared Euclidean Distance
 * TWE

Our implementations have a O(n) space complexity and tightly integrate pruning with early abandoning,
providing high performance and scalability.
They should provide a significant speed up in most scenarios, even though there worst time complexity remain O(n²).
Note that they will not always be faster than the naive implementations
as early abandoning and pruning require some bookkeeping.
This bookkeeping must be compensated for, and this is not always possible
(e.g. when the series are too short, with windows too small).

In C++/Python, we have two versions of each distance: one with a cut-off point and one without
(In java, only the cut-off version is available).
The cut-off point is an upper bound above which pruning starts.
When "enough" pruning occurs (i.e. a full line of the cost matrix), the computation is early abandoned.
When no cut-off point is provided, we compute one ourselves
(based one the diagonal and the last column of the cost matrix)
before calling the version with cut-off.
This allows to prune computation even when no cut-off point is provided
(no early abandoning can occur under these conditions).

### Benckmarks
The early abandoning and pruning algorithm is submitted in a paper under review.
The paper ([pre-print pdf](https://arxiv.org/abs/2102.05221))
contains benchmarks that can be found here https://github.com/HerrmannM/paper-2021-EAPElasticDist.


## Generic approach to "transforms" (C++)
The derivative version of DTW, CDTW and WDTW are not directly implemented.
We rather use the concept of "transforms", which is a pre-processing step.
Currently available tranforms are:
 * Upper and Lower envelope computations (to use with lower bound)
 * Derivative (can be repeated)

Hence, you can easily set up a derivative (or nth derivative) of any distance.


## Proximity Forest library (C++)



# Using Tempo in your project

## As a C++ library with CMake
Download the sources and link to Tempo adding in your CMakeLists.txt:
```cmake
add_subdirectory(path_to_tempo)
target_link_libraries(<target> PRIVATE tempo)
```

## Python bindings
Bindings tested for Python3. All commands here assume a Python3 environment.
Ensure that the `wheel` packages is installed (e.g. with `pip install wheel`)
```bash
pip install wheel                             # To install wheel
python setup.py bdist_wheel                   # Build the package
pip install ./dist/pytempo-[info].whl         # Path to wheel ( --force-reinstall to overwrite a previous installation)
```

Distances must be called with numpy arrays.
Have a look at the example [bindings/python/example.py](bindings/python/example.py).

To remove the package:
```bash
pip uninstall pytempo.distances
```

Python bindings are made with [pybind11](https://github.com/pybind/pybind11).

## As a Java library
Simply copy/past the file from the [java directory](./bindings/java).


# Applications
Our applications are not only demonstrators: you can use them for your work too!
The only constraint is to use the TS format from
[timeseriesclassification.com](http://timeseriesclassification.com/dataset.php) for your data.
You can either give the path to a UCR Archive folder and dataset name (flag `-ucr`),
or directly the train and test files (flag `-tt`).

## Building an applications
**Note**: we developed and tested our applications under Linux, using standard C++ 17.
The following steps work in a Linux terminal, and should work on MacOS too.
For Windows, refer to CMake user doc.

Open a terminal, and type in the following command, `<app-folder>` being the path to the desired application's folder,
and `<app-name>` its name which is also the executable name (e.g. `nn1dist`).
The last line will print the help message of the applications.

```bash
cd <app-folder>
mkdir cmake-build-release
cd cmake-build-release
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make
./<app-name>
```

## [NN1](./app/nn1dist)
C++ NN1 classification using one of the elastic distances.
The distance is specified with the ``-dist`` flag, followed by the distance arguments (see the help message).
A nth derivative can be specified with the ``-tr derivative n`` flag.
So
```bash
./nn1dist -dist msm 0.05 -ucr ~/Univariate_ts/ Crop -tr derivative 1
```
will compute a "Derivative MSM".

## [Proximity Forest 2018](./app/pf2018)


# About
Tempo is a project of the [Time Series Classification team from Monash University](https://www.monash.edu/it/dsai/machine-learning).
It draws inspiration from many scientific sources that can be found in the [reference folder](./doc/references).
Tempo is developed by [Dr. Matthieu Herrmann](https://github.com/HerrmannM)