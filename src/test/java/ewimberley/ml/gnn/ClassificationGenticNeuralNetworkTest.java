package ewimberley.ml.gnn;

import org.junit.Test;

import ewimberley.ml.DataLoader;
import ewimberley.ml.Learner;

public class ClassificationGenticNeuralNetworkTest {

	@Test
	public void testIrisData() {
		String dataFile = "src/test/resources/iris.data";
		DataLoader dl = new DataLoader();
		dl.loadCSVFile(dataFile);
		Learner bestNetwork = ClassificationGenticNeuralNetwork.train(dl.getData(), dl.getClassLabels(), 1000, 500, 5, 8, 100.0);
	}

}
