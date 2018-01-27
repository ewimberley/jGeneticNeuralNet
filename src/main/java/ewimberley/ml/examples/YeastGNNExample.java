package ewimberley.ml.examples;

import ewimberley.ml.DataLoader;
import ewimberley.ml.ann.gnn.classifier.ClassificationGenticNeuralNetwork;

public class YeastGNNExample {

	public static void main(String[] args) {
		/*
		 * Lichman, M. (2013). UCI Machine Learning Repository
		 * [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School
		 * of Information and Computer Science.
		 */
		String dataFile = "src/test/resources/yeast.data";
		DataLoader dl = new DataLoader();
		dl.loadCSVFile(dataFile);
		ClassificationGenticNeuralNetwork model = ClassificationGenticNeuralNetwork.train(dl.getData(),
				dl.getClassLabels(), 500, 1000, 5, 8, 100.0);
		model.printNetwork();
	}

}
