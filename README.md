# Tempo ![logo](./doc/logos/tempo100.png)
Monash Time Series Analytics Library.
Monash University, Melbourne, Australia.

Tempo is not only a project for researchers!
Our aim is to deliver a user-friendly library,
allowing anyone to enjoy our last work in the domain of time series classification.

**Note:** Tempo is young and in active development. Do not hesitate to report issues!

**Note:** We experiment using the UCR archive, mostly in its 85 datasets version.
The archive can be found at [timeseriesclassification.com](http://timeseriesclassification.com/dataset.php).

**Note:** The previous version `v0.0.1` can be found in the tags.
The current version does not yet contain all the code and applications that use to be there.
In particular, the Java implementations of the distances are in the v0.0.1.
If you need them, check the tags!

If you are using tempo in your research, please cite
```
@article{Herrmann2021,
author={Herrmann, Matthieu and Webb, Geoffrey I.},
title={Early abandoning and pruning for elastic distances including dynamic time warping},
journal={Data Mining and Knowledge Discovery},
year={2021},
month={Nov},
day={01},
volume={35},
number={6},
pages={2577-2601},
doi={10.1007/s10618-021-00782-4},
url={https://doi.org/10.1007/s10618-021-00782-4}
}
```

# Features

## Elastic distances (C++, Python3)
Starting point of our library, we provide implementations of the most commonly used elastic distances.
Our implementations in C++ (and Java, available in v0.0.1) are early abandoned and pruned, and we offer [Python3 bindings](#python3-bindings).
Available distances are
 * ADTW (see https://arxiv.org/abs/2111.13314)
 * DTW and CDTW, with lower bounds (Lb Keogh, Lb Enhanced, Lb Webb)
 * WDTW
 * Direct Alignment (like the squared Euclidean Distance)
 * ERP
 * LCSS
 * MSM
 * TWE

Our implementations have a O(n) space complexity and tightly integrate pruning with early abandoning,
providing high performance and scalability.
They should provide a significant speed up in most scenarios, even though there worst time complexity remain O(n²).
Note that they will not always be faster than the naive implementations
as early abandoning and pruning require some bookkeeping.
This bookkeeping must be compensated for, and this is not always possible
(e.g. when the series are too short, with windows too small).

The distances accept a cut-off point, an upper bound above which pruning starts.
When "enough" pruning occurs (i.e. a full line of the cost matrix),
the computation is early abandoned.
When no cut-off point is provided, we compute one ourselves
(based one the diagonal and the last column of the cost matrix)
before calling the version with cut-off.
This allows to prune computation even when no cut-off point is provided
(no early abandoning can occur under these conditions).

### Benchmarks
The [published paper](https://link.springer.com/article/10.1007/s10618-021-00782-4)
contains benchmarks that can be found [here](https://github.com/HerrmannM/paper-2021-EAPElasticDist).


# Using Tempo in your project

## As a C++ library with CMake
Download the sources and link to Tempo adding in your CMakeLists.txt:
```cmake
add_subdirectory(path_to_tempo)
target_link_libraries(<your target> PRIVATE tempo)
```

## Python bindings
Bindings tested for Python3. All commands here assume a Python3 environment.
Ensure that the `wheel` packages is installed (e.g. with `pip install wheel`).
It is recommended to use a virtual environment.

The following command are assumed to be typed in the `extra/python/test` directory.
```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

And this commands from the `extra/python` directory.
The commands must be modified according to your system.
In particular, the name of the created wheel will be different,
but will always be in the `dist`
directory.
```bash
python3 setup.py bdist_wheel
pip install dist/pytempo-0.0.2-cp310-cp310-linux_x86_64.whl # To adapt!
pytest -s test 
```

Distances must be called with numpy arrays.
Have a look at the example [extra/python/test/example.py](bindings/python/example.py).

Python bindings are made with [pybind11](https://github.com/pybind/pybind11).

## Note:
Warping windows w are commonly expressed as a ratio of the length of the series, e.g. `w=0.1`.
In tempo, the window parameter is the actual window size, e.g. `w=14`.

# About
Tempo is a project of the [Time Series Classification team from Monash University](https://www.monash.edu/it/dsai/machine-learning).
It draws inspiration from many scientific sources that can be found in the [reference folder](./doc/references).
Tempo is developed by [Dr. Matthieu Herrmann](https://github.com/HerrmannM)
