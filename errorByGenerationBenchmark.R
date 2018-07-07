library(ggplot2)
options(echo=TRUE)
args <- commandArgs(trailingOnly = TRUE)
benchmarkDataFile <- args[1]
ymax <- args[2]
performanceData <- read.csv(benchmarkDataFile, header = TRUE)
png(file = "ErrorByGeneration.png")
colnames(performanceData)[1] <- "NumGenerations"
colnames(performanceData)[2] <- "MeanSquaredError"
ggplot(performanceData, aes(x=NumGenerations, y=MeanSquaredError)) + geom_boxplot(aes(group = cut_width(NumGenerations, 15.0))) + scale_y_continuous(limits=c(0,as.numeric(ymax)))
dev.off()

