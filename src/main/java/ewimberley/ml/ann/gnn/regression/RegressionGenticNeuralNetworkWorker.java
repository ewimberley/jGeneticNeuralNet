package ewimberley.ml.ann.gnn.regression;

import java.util.List;

import ewimberley.ml.ann.gnn.GeneticNeuralNetworkWorker;

/**
 * A thread that trains regression networks.
 * 
 * @author ewimberley
 *
 */
public class RegressionGenticNeuralNetworkWorker
		extends GeneticNeuralNetworkWorker<Double, RegressionGenticNeuralNetwork> {

	public RegressionGenticNeuralNetworkWorker(RegressionGenticNeuralNetwork network, double[][] data, Double[] y,
			List<Integer> trainingIndices) {
		super(network, data, y, trainingIndices);
	}

	public void train() {
		mutant = new RegressionGenticNeuralNetwork(original);
		mutant.mutate();
		computeAverageError();
	}

}
