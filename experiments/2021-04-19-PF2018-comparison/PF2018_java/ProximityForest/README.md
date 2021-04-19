

# ProximityForest
An effective and scalable distance-based classifier for time series classification. This repostitory contains the source code for the time series classification algorithm Proximity Forest, published in the paper [https://arxiv.org/abs/1808.10594]

## Abstract 
Research into the classification of time series has made enormous
progress in the last decade. The UCR time series archive has played a significant
role in challenging and guiding the development of new learners for time
series classification. The largest dataset in the UCR archive holds 10 thousand
time series only; which may explain why the primary research focus has been
on creating algorithms that have high accuracy on relatively small datasets.
This paper introduces Proximity Forest, an algorithm that learns accurate
models from datasets with millions of time series, and classifies a time series in
milliseconds. The models are ensembles of highly randomized Proximity Trees.
Whereas conventional decision trees branch on attribute values (and usually
perform poorly on time series), Proximity Trees branch on the proximity of
time series to one exemplar time series or another; allowing us to leverage
the decades of work into developing relevant measures for time series. Proximity
Forest gains both efficiency and accuracy by stochastic selection of both
exemplars and similarity measures.

Our work is motivated by recent time series applications that provide orders
of magnitude more time series than the UCR benchmarks. Our experiments
demonstrate that Proximity Forest is highly competitive on the UCR archive:
it ranks among the most accurate classifiers while being significantly faster.
We demonstrate on a 1M time series Earth observation dataset that Proximity
Forest retains this accuracy on datasets that are many orders of magnitude
greater than those in the UCR repository, while learning its models at least
100,000 times faster than current state of the art models Elastic Ensemble and
COTE.

When using this repository, please cite:
```
@Article{Lucas2019,
  author="Lucas, Benjamin and Shifaz, Ahmed and Pelletier, Charlotte and O'Neill, Lachlan and Zaidi, Nayyar and Goethals, Bart and Petitjean, Fran{\c{c}}ois and Webb, Geoffrey I.",
  title="Proximity Forest: an effective and scalable distance-based classifier for time series",
  journal="Data Mining and Knowledge Discovery",
  year="2019",
  doi="10.1007/s10618-019-00617-3"
}

```

## Usage and prerequisites

The project requires Java 8.0, and two open source libraries [Apache Commons Lang](https://commons.apache.org/proper/commons-lang/) 3.7, and [Google Gson](https://github.com/google/gson) 2.8.2. These two libraries are included in the lib folder.

The project was developed using Eclipse 4.8 IDE. If you require the project to be moved to another IDE, just create a new project and import src and lib directories to the new IDE.

###  Creating a jar file

Open the project in Eclipse and use File->Export->Java->JAR File. to export a `.jar` file of the project

### Input data format
This implementation currently supports CSV files with the following format for testing and training data. 

 - A matrix of comma separated double values
 - Class label is either the first or last column, refer to the command line options below
 - If header row is included, refer to the command line options below
 - All time series are expected to have the same length 
 - Data z-normalised per series works best  - TODO

### Running in command line 
```
java -jar -Xmx1g ProximityForest.jar 
-train=E:/data/ucr/ItalyPowerDemand/ItalyPowerDemand_TRAIN.csv 
-test=E:/data/ucr/ItalyPowerDemand/ItalyPowerDemand_TEST.csv 
-out=output -repeats=1 -trees=100 -r=1 -on_tree=true -export=1 -verbosity=0
```
The `Xmx1g` sets the Java Virtual Machine memory limit to 1GB - increase this value as required by your dataset. Use the `-verbosity` option to print memory usage.

Separate each command line option with an `=`. Available options are 
- `-train`=path to training file (only CSV files are supported)
- `-test`=path to training file (only CSV files are supported)
- `-out`=output folder, folder is created if it does not exist
- `-repeats`=number of times to repeat the experiment, default is 1, but since this is a randomised ensemble it is recommended to repeat the experiment at last 10 times. Especially if the number of trees is low
- `-trees`=number of trees in the ensemble, the default is 1 to run an initial test quickly, but you can start testing with 10, 20, 50, 100, etc.
- `-r`= number of candidate splits to evaluate per node, if not specified the default is 1, recommended value is 5, but note that this will increase the training time approximately 5 times, but will give a higher accuracy
- `-on_tree`= if `true` distance measure is selected per node, if `false` it is selected once per tree. , if not specified the default is `true` 
- `-shuffle`= if `true` shuffles the training dataset, if not specified the default is `false` 
<!---
- `-jvmwarmup`= if `true` some extra calculation is done before the experiment is started to "warmup" java virtual machine, this helps measure more accurate elapsed time for short duration.
-->
- `-export`= set to 1 to export results in json format to the specified output directory, if 0 results are not exported.
- `-verbosity`= if 0 minimal printing to stdout, if 1 progress is printed per tree, if 2 memory usage is printed per tree, the default is 0
- `-csv_has_header`= set to `true` if input csv files contain a header row, default assumes `false`
- `-target_column`= set `first` if first column contains target label in the input files, or set to `last` if the last column is the target label. The default is `first`, and only the values `first` and `last` is supported.  

If an option is not specified in the command line, a default setting is used, which can be found in the `AppContext.java`class.

# Support

Some classes in the package distance.elastic may contain borrowed code from the timeseriesweka project at the www.timeseriesclassification.com (Bagnall, 2017).  We have modified the original classes to fix bugs and improve the implementation efficiency. Their paper "The great time series classification bakeoff " (Bagnall, 2017) has been cited in our Proximity Forest (B.Lucas, 2018) paper as well.

YourKit is supporting Proximity Forest open source project with its full-featured Java Profiler.
YourKit is the creator of innovative and intelligent tools for profiling Java and .NET applications. http://www.yourkit.com 

