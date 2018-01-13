package ewimberley.ml.ann.gnn.classifier;

import org.junit.Test;

import ewimberley.ml.DataLoader;
import ewimberley.ml.Learner;
import ewimberley.ml.ann.gnn.classifier.ClassificationGenticNeuralNetwork;

public class ClassificationGenticNeuralNetworkTest {

	@Test
	public void testIrisData() {
		String dataFile = "src/test/resources/iris.data";
		DataLoader dl = new DataLoader();
		dl.loadCSVFile(dataFile);
		Learner bestNetwork = ClassificationGenticNeuralNetwork.train(dl.getData(), dl.getClassLabels(), 1000, 500, 5, 8, 100.0);
	}

}
