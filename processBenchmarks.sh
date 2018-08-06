#!/bin/sh
Rscript errorByGenerationBenchmark.R xsquared_benchmark.csv 3
mv ErrorByGeneration.png images
Rscript accuracyByGenerationBenchmark.R iris_benchmark.csv 0.4
mv AccuracyByGeneration.png images
