package ewimberley.ml.ann.gnn.regression;

import java.util.List;

public class RegressionGenticNeuralNetworkWorker implements Runnable {

	private RegressionGenticNeuralNetwork original, mutant;

	private double[][] data;

	private Double[] y;

	private List<Integer> trainingIndices;

	private double error;

	public RegressionGenticNeuralNetworkWorker(RegressionGenticNeuralNetwork network, double[][] data, Double[] y, List<Integer> trainingIndices) {
		this.original = network;
		this.data = data;
		this.trainingIndices = trainingIndices;
		this.y = y;
	}

	public void run() {
		mutant = new RegressionGenticNeuralNetwork(original);
		mutant.mutate();
		double averageOriginalError = 0.0;
		double averageMutantError = 0.0;
		for (Integer trainingIndex : trainingIndices) {
			if (original.getAverageError() == -1.0) {
				//this may be memoized
				averageOriginalError += original.error(data[trainingIndex], y[trainingIndex]);
			}
			averageMutantError += mutant.error(data[trainingIndex], y[trainingIndex]);
		}
		// calculate original average error
		if (original.getAverageError() == -1.0) {
			averageOriginalError /= trainingIndices.size();
			original.setAverageError(averageOriginalError);
		}
//		System.out.println("Original avg error is: " + original.getAverageError());
		// calculate mutant average error
		averageMutantError /= trainingIndices.size();
//		System.out.println("Mutant avg error is: " + averageMutantError);
		mutant.setAverageError(averageMutantError);
	}

	public double getError() {
		return error;
	}

	public RegressionGenticNeuralNetwork getOriginal() {
		return original;
	}

	public RegressionGenticNeuralNetwork getMutant() {
		return mutant;
	}

}
