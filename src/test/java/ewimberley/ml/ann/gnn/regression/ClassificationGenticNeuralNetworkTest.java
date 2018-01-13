package ewimberley.ml.ann.gnn.regression;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Ignore;
import org.junit.Test;

import ewimberley.ml.DataLoader;

public class ClassificationGenticNeuralNetworkTest {

	@Test
	public void testParabolaData() {
		int numX = 100;
		double[][] data = new double[numX][1];
		double[] values = new double[numX];
		for(int i = 0; i < numX; i++) {
			data[i][0] = i;
			values[i] = i*i;
		}
		RegressionGenticNeuralNetwork bestNetwork = (RegressionGenticNeuralNetwork) RegressionGenticNeuralNetwork
				.train(data, values, 1000, 500, 5, 8, 100.0);
		assertTrue(bestNetwork.getAverageError() < 1.0);
	}

//	@Test
//	public void testShortIrisData() {
//		String dataFile = "src/test/resources/iris.data";
//		DataLoader dl = new DataLoader();
//		dl.loadCSVFile(dataFile);
//		ClassificationGenticNeuralNetwork bestNetwork = (ClassificationGenticNeuralNetwork) ClassificationGenticNeuralNetwork
//				.train(dl.getData(), dl.getClassLabels(), 10, 10, 5, 8, 100.0);
//		assertTrue(bestNetwork.getAverageError() <= 3.0);
//	}
//
//	@Test
//	public void testActivationSum() {
//		double[][] data = new double[2][1];
//		data[0][0] = 0.0;
//		data[1][0] = 1.0;
//		String[] labels = new String[2];
//		labels[0] = "Test";
//		labels[1] = "Test2";
//		ClassificationGenticNeuralNetwork bestNetwork = (ClassificationGenticNeuralNetwork) ClassificationGenticNeuralNetwork
//				.train(data, labels, 1, 1, 1, 1, 100.0);
//		assertEquals(1, bestNetwork.getInputs().size());
//		assertEquals(2, bestNetwork.getOutputs().size());
//		assertEquals(3, bestNetwork.getLayers().size());
//		assertEquals(4, bestNetwork.getNeurons().keySet().size());
//	}

}
