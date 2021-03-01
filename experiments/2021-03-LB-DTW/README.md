# (C)DTW Lower Bound Experiments
Create: 24/02/2021

## Introduction
In the context of NN1 classification under DTW and CDTW,
we want to classify a query Q given a database of candidates C_i:
the class of Q is the class of the nearest (smallest distance) candidate C_i.
The process can be accelerated using lower bounds `LB`.
As `LB(Q, C_i)<=DTW(Q, C_i)` (and similarly for CDTW),
we do not need to compute the full distance if `LB(Q, C_i)`
is already greater than the distance of the current best candidate.
To be effective, a lower bound must be "tight" (as close as possible to the distance)
while remaining computationaly cheap.

All the implemented lower bounds relies on some data derived from the series
(in the present case, different kind of "envelopes").
In this experiment, we use our [nn1dist](../../app/nn1dist) application,
which always pre-compute all the required information.
The database being heavily used, precomputing the information for its series is common.
On the query side, these data are usually computed on-demand, if at all.
If all the queries are known in advance, they can also be pre-computed.
We choose to do so, although the benefit is more about code architecture rather than speed
(it actually requires more memory than an on-demand approach).
Note that the cost of pre-computing the envelopes is negligible compared to the rest,
thanks to Daniel Lemire's algorithm.

## Experimental Setup
In this experiment, we assess the efficiency of several lower bounds.
We use the UCR Archive in its 85 univariate datasets version, with both DTW and CDTW.
For CDTW, we use the best window found by EE.
We only consider datasets with same-length series.

We currently (24/02/2021) have the following lower bounds implemented:
* `lb-keogh`: The usual LB-Keogh, requiring the envelopes of one of Q or C.
  We precompute the envelopes of the database.
* `lb-keogh2`: LB-Keogh is not symmetric, i.e. in general `lb-keogh(Q, C) != lb-keogh(C, Q)`.
  Hence, we can "cascade" them: do one of them, and if it does not allow to stop, do the other.
* `lb-enhanced`: Requires a number of bands `v`. The first and last `v` bands are tighter than LB-Keogh.
  Theses bands are then bridged by `lb-keogh`. LB-Enhanced with `v=1` is strictly tighter than LB-Keogh.
  Higher `v` usually gives tighter results, although not always.
* `lb-webb`: The tightest lower bound known today, but requires 8 different envelopes.

We also added a variation for `lb-keogh2` and `lb-enhanced`:
* `lb-keogh2j`: Doing both LB-Keogh together ('joined') rather than cascading.
* `lb-enhanced2j`: Same as `lb-enhanced` but bridging with `lb-keogh2j`.

The experiments are run on a Linux system.

## Description of the files

* [eeOutputFold0.csv](./eeOutputFold0.csv) contains the best parameters found by EE for several elastic distances.
  We are only interest in the window for CDTW.
* [generatecmds.py](./generatecmds.py) is a python script generating command lines.
* [parlauncher.py](./parlauncher.py) is a pyhton script reading commands from a file and starting them in parallel.
