package ewimberley.gnn;

import org.junit.Test;

public class GeneticNeuralNetTest {

	@Test
	public void testIrisData() {
		String dataFile = "src/test/resources/iris.data";
		DataLoader dl = new DataLoader();
		dl.loadCSVFile(dataFile);
		Classifier gnn = new GenticNeuralNetwork(dl.getData(), dl.getClassLabels());
		Classifier bestNetwork = GenticNeuralNetwork.train(dl.getData(), dl.getClassLabels(), 500, 500, 5, 8, 100.0);
	}

}
