library(ggplot2)
options(echo=TRUE)
args <- commandArgs(trailingOnly = TRUE)
benchmarkDataFile <- args[1]
ymin <- args[2]
performanceData <- read.csv(benchmarkDataFile, header = TRUE)
png(file = "AccuracyByGeneration.png")
colnames(performanceData)[1] <- "NumGenerations"
colnames(performanceData)[2] <- "Accuracy"
ggplot(performanceData, aes(x=NumGenerations, y=Accuracy)) + geom_boxplot(aes(group = cut_width(NumGenerations, 15.0))) + scale_y_continuous(limits=c(as.numeric(ymin),1.05))
dev.off()

