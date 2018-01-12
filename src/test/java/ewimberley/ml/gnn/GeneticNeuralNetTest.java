package ewimberley.ml.gnn;

import org.junit.Test;

import ewimberley.ml.Classifier;
import ewimberley.ml.DataLoader;
import ewimberley.ml.gnn.GenticNeuralNetwork;

public class GeneticNeuralNetTest {

	@Test
	public void testIrisData() {
		String dataFile = "src/test/resources/iris.data";
		DataLoader dl = new DataLoader();
		dl.loadCSVFile(dataFile);
		Classifier gnn = new GenticNeuralNetwork(dl.getData(), dl.getClassLabels());
		Classifier bestNetwork = GenticNeuralNetwork.train(dl.getData(), dl.getClassLabels(), 1000, 500, 5, 8, 100.0);
	}

}
