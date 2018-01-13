package ewimberley.ml.gnn;

import java.util.List;

public class NeuralNetworkWorker<H> implements Runnable {

	private GenticNeuralNetwork<H> original, mutant;

	private double[][] data;

	private String[] classLabels;

	private List<Integer> trainingIndices;

	private double error;

	public NeuralNetworkWorker(GenticNeuralNetwork<H> network, double[][] data, String[] labels, List<Integer> trainingIndices) {
		this.original = network;
		this.data = data;
		this.trainingIndices = trainingIndices;
		this.classLabels = labels;
	}

	public void run() {
		mutant = new GenticNeuralNetwork<H>(original);
		mutant.mutate();
		double averageOriginalError = 0.0;
		double averageMutantError = 0.0;
		for (Integer trainingIndex : trainingIndices) {
			if (original.getAverageError() == -1.0) {
				//this may be memoized
				averageOriginalError += original.error(data[trainingIndex], classLabels[trainingIndex]);
			}
			averageMutantError += mutant.error(data[trainingIndex], classLabels[trainingIndex]);
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

	public GenticNeuralNetwork<H> getOriginal() {
		return original;
	}

	public GenticNeuralNetwork<H> getMutant() {
		return mutant;
	}

}
