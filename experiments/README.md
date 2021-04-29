# Experiments

Our experiments are run on Linux, and expect a compatible environment (MacOS may be ok).
In particular, we generate command line suitable for Linux system, which probably won't work on Windows.


## Dataset configuration
Experiments rely on datasets from the UCR archive in TS format
(see [http://timeseriesclassification.com/dataset.php](http://timeseriesclassification.com/dataset.php)).
Due to their size, they are better left outside of this repository.
Use the file [CONFIGURE_ME.py](./CONFIGURE_ME.py) to points to the archive directory.
Experience script will us this file to find the datasets.

### Note on the "UCR archive version"
We mentioned several time the use of the UCR archive in "its 85 datasets version".
The UCR archive you can download now of March 2021 has 128 datasets, but it used to have 85.
The Elastic Ensemble EE, which produces optimal parametrization of distances per datasets.
For now, we only have its results for the first 85 datasets, hence the mention.


## Description of the files
* [CONFIGURE_ME.py](./CONFIGURE_ME.py) Tell us where is you UCR archive (in TS format) is located. 
* [eeOutputFold0.csv](./eeOutputFold0.csv) Best parameters found by EE for 11 elastic distances. 
  For now (March 2021), in its 85 datasets version (UCR archive).
* [ee_fold0_accuracies.json](./ee_fold0_accuracies.json) and [ee_fold0_accuracires.csv](./ee_fold0_accuracies.csv)
  Accuracies of the 11 EE elastic distances using the EE best parameters.
* [parlauncher.py](./parlauncher.py) Pyhton script reading commands from a file and starting them in parallel.
