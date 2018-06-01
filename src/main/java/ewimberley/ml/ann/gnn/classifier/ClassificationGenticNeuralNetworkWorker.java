package ewimberley.ml.ann.gnn.classifier;

import java.util.List;

import ewimberley.ml.ann.gnn.GeneticNeuralNetworkWorker;

/**
 * A thread that trains classification networks.
 * 
 * @author ewimberley
 *
 */
public class ClassificationGenticNeuralNetworkWorker
		extends GeneticNeuralNetworkWorker<String, ClassificationGenticNeuralNetwork> {

	public ClassificationGenticNeuralNetworkWorker(ClassificationGenticNeuralNetwork network, double[][] data,
			String[] labels, List<Integer> trainingIndices) {
		super(network, data, labels, trainingIndices);
	}

	public void train() {
		mutant = new ClassificationGenticNeuralNetwork(original);
		mutant.mutate();
		computeAverageError();
	}

}
