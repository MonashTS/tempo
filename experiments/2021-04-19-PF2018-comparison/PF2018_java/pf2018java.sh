#!/bin/bash
BASEDIR=$(dirname "$0")
java -jar -Xmx4g ${BASEDIR}/pf2018_java.jar $*
