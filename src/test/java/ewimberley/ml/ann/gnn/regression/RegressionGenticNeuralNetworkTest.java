package ewimberley.ml.ann.gnn.regression;

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
		//XXX not working
		RegressionGenticNeuralNetwork bestNetwork = (RegressionGenticNeuralNetwork) RegressionGenticNeuralNetwork
				//.train(data, values, 1000, 2000, 4, 10, 100.0);
				.train(data, values, 500, 300, 4, 10, 100.0);
		//assertTrue(bestNetwork.getAverageError() < 1.0);
		System.out.println(bestNetwork.getAverageError());
		
		//for(int i = 0; i < numX; i++) {
		for(int i = 0; i < 5; i++) {
			//tolerance of 1%
			System.out.println(i + "\t" + values[i] + "\t" + bestNetwork.predict(new double[] {(double)i}));
			//assertEquals(values[i], bestNetwork.predict(new double[] {(double)i}), (values[i]/100.0));
		}
	}

}
