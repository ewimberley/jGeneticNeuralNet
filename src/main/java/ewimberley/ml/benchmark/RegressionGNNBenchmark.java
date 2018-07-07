package ewimberley.ml.benchmark;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import ewimberley.ml.ann.gnn.GeneticNeuralNetworkTrainingConfiguration;
import ewimberley.ml.ann.gnn.regression.RegressionGenticNeuralNetwork;

public class RegressionGNNBenchmark {

	private static final int MAX_PER_INCREMENT = 400;
	private static final int MAX_GEN = 500;
	private static final int MIN_GEN = 15;

	public RegressionGNNBenchmark() {

	}

	public static void main(String[] args) {
		RegressionGNNBenchmark main = new RegressionGNNBenchmark();
		main.train();
	}

	private void train() {
		int numX = 100;
		double[][] data = new double[numX][1];
		double[] values = new double[numX];
		for (int i = 0; i < numX; i++) {
			data[i][0] = i;
			values[i] = i * i;
		}
		GeneticNeuralNetworkTrainingConfiguration config = new GeneticNeuralNetworkTrainingConfiguration();
		config.setNumNetworksPerGeneration(1000);
		config.setNumHiddenLayers(2);
		config.setNumNeuronsPerLayer(6);
		config.setMaxLearningRate(1.0);
		config.setProbMutateActivationFunction(0.1);
		config.setMaxThreads(Runtime.getRuntime().availableProcessors() * 2);
		
		try {
			FileWriter fileWriter = new FileWriter("/home/ewimberley/xsquared_benchmark.csv");
			PrintWriter printWriter = new PrintWriter(fileWriter);
			printWriter.print("numGenerations,meanSquaredError\n");
			for (int numGens = MIN_GEN; numGens <= MAX_GEN; numGens *= 2) {
				config.setNumGenerations(numGens);
				for (int i = 0; i < MAX_PER_INCREMENT; i++) {
					config.setNumGenerations(numGens);
					RegressionGenticNeuralNetwork bestNetwork = (RegressionGenticNeuralNetwork) RegressionGenticNeuralNetwork
							.train(data, values, config);
					printWriter.print(numGens + ", " + bestNetwork.getAverageError() + "\n");
				}
			}
			printWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
