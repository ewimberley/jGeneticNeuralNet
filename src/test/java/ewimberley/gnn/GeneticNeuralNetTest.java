package ewimberley.gnn;

import org.junit.Test;

public class GeneticNeuralNetTest {

	@Test
	public void testIrisData() {
		String dataFile = "src/test/resources/iris.data";
		DataLoader dl = new DataLoader();
		dl.loadCSVFile(dataFile);
		NeuralNetwork gnn = new GenticNeuralNetwork(dl.getData(), dl.getClassLabels());
		NeuralNetwork bestNetwork = GenticNeuralNetwork.train(dl.getData(), dl.getClassLabels(), 100, 1000, 5, 8, 100.0);
	}

}
