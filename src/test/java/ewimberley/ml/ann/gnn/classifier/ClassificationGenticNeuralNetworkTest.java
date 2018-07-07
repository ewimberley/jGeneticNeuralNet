package ewimberley.ml.ann.gnn.classifier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import ewimberley.ml.DataLoader;
import ewimberley.ml.ann.gnn.GenticNeuralNetwork;
import ewimberley.ml.ann.gnn.GeneticNeuralNetworkTrainingConfiguration;

public class ClassificationGenticNeuralNetworkTest {

	@Test
	public void testShortIrisData() {
		/*
		 * Lichman, M. (2013). UCI Machine Learning Repository
		 * [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School
		 * of Information and Computer Science.
		 */
		String dataFile = "src/test/resources/iris.data";
		DataLoader dl = new DataLoader();
		dl.loadCSVFile(dataFile);
		GeneticNeuralNetworkTrainingConfiguration config = new GeneticNeuralNetworkTrainingConfiguration();
		config.setNumNetworksPerGeneration(100);
		config.setNumGenerations(1000);
		config.setNumHiddenLayers(5);
		config.setNumNeuronsPerLayer(12);
		config.setMaxLearningRate(100.0);
		GenticNeuralNetwork<String> bestNetwork = (GenticNeuralNetwork<String>) ClassificationGenticNeuralNetwork
				.train(dl.getData(), dl.getClassLabels(), config);
		assertTrue(bestNetwork.getAverageError() <= 3.0);
	}

	@Test
	public void testActivationSum() {
		double[][] data = new double[2][1];
		data[0][0] = 0.0;
		data[1][0] = 1.0;
		String[] labels = new String[2];
		labels[0] = "Test";
		labels[1] = "Test2";
		GeneticNeuralNetworkTrainingConfiguration config = new GeneticNeuralNetworkTrainingConfiguration();
		config.setNumNetworksPerGeneration(100);
		config.setNumGenerations(1000);
		config.setNumHiddenLayers(5);
		config.setNumNeuronsPerLayer(12);
		config.setMaxLearningRate(100.0);
		GenticNeuralNetwork<String> bestNetwork = (GenticNeuralNetwork<String>) ClassificationGenticNeuralNetwork
				.train(data, labels, config);
		assertEquals(1, bestNetwork.getInputs().size());
		assertEquals(2, bestNetwork.getOutputs().size());
		assertEquals(3, bestNetwork.getLayers().size());
		assertEquals(4, bestNetwork.getNeurons().keySet().size());
	}

}
