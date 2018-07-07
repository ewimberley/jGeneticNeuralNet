package ewimberley.ml.benchmark;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import ewimberley.ml.ConfusionMatrix;
import ewimberley.ml.DataLoader;
import ewimberley.ml.ann.NeuralNetworkTrainingConfiguration;
import ewimberley.ml.ann.gnn.GenticNeuralNetwork;
import ewimberley.ml.ann.gnn.classifier.ClassificationGenticNeuralNetwork;

public class IrisGNNBenchmark {

	private static final int MAX_PER_INCREMENT = 200;
	private static final int MAX_GEN = 500;
	private static final int MIN_GEN = 15;

	public IrisGNNBenchmark() {

	}

	public static void main(String[] args) {
		IrisGNNBenchmark main = new IrisGNNBenchmark();
		main.train();
	}

	private void train() {
		String dataFile = "src/test/resources/iris.data";
		DataLoader dl = new DataLoader();
		dl.loadCSVFile(dataFile);

		NeuralNetworkTrainingConfiguration config = new NeuralNetworkTrainingConfiguration();
		config.setNumNetworksPerGeneration(3000);
		config.setNumHiddenLayers(3);
		config.setNumNeuronsPerLayer(8);
		config.setMaxLearningRate(10.0);

		try {
			FileWriter fileWriter = new FileWriter("/home/ewimberley/iris_benchmark.csv");
		    PrintWriter printWriter = new PrintWriter(fileWriter);
		    printWriter.print("numGenerations,accuracy\n");
			for (int numGens = MIN_GEN; numGens <= MAX_GEN; numGens *= 2) {
				config.setNumGenerations(numGens);
				for (int i = 0; i < MAX_PER_INCREMENT; i++) {
					GenticNeuralNetwork<String> model = ClassificationGenticNeuralNetwork.train(dl.getData(),
							dl.getClassLabels(), config);
					ConfusionMatrix cf = model.getConfusion();
					printWriter.print(numGens + ", " + cf.computeAccuracy() + "\n");
				}
			}
			printWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
