package ewimberley.ml.ann.gnn.regression;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class RegressionGenticNeuralNetworkTest {

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
				.train(data, values, 50, 50, 2, 5, 1000.0);
		assertTrue(bestNetwork.getAverageError() < 1.0);
	}

}
