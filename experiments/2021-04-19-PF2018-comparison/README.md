# Runtime PF 2018 Java vs CPP
This experiment compares the execution time of our reimplementation
of Proximity Forest in C++ using our early abandoned and pruned distances
(see the folder [app/pf2018](../../app/pf2018))
against the [java implementation from 2018](https://github.com/fpetitjean/ProximityForest).

Note that this comparison is not suitable to evaluate the impact of pruning and early abandoning
in Proximity Forest, as the change of language and overall different implementation also have an
impact on runtime. However, it gives an idea of how fast the system can be.

## The java version
We modified the CSV parser of the java version, allowing it to read univariate series from arff files.
It simply ignores the lines starting by '%', '@' and the empty lines.
The updated source of the java version can be found in the [PF2018_java](PF2018_java) folder.
This folder also contained the jar file of the project.
