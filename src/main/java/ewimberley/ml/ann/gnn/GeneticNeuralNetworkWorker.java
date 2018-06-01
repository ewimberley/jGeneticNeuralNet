package ewimberley.ml.ann.gnn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * The parent class for the thread that trains a network.
 * 
 * @author ewimberley
 *
 * @param <Y>
 *            the output type
 * @param <N>
 *            the network type
 */
public abstract class GeneticNeuralNetworkWorker<Y, N extends GenticNeuralNetwork<Y>> implements Runnable {

	//XXX make this a hyperparameter
	//FIXME increase this over time?
	private static final double TRAINING_SAMPLE_RATIO = 0.5;
	
	protected N original;
	protected N mutant;
	protected double[][] data;
	protected List<Integer> trainingIndices;
	protected double error;
	protected Y[] y;

	public GeneticNeuralNetworkWorker(N network, double[][] data,
			Y[] labels, List<Integer> trainingIndices) {
		this.original = network;
		this.data = data;
		this.trainingIndices = trainingIndices;
		this.y = labels;
	}

	/**
	 * Run in a new thread.
	 */
	public void run() {
		train();
	}

	/**
	 * Custom training function implemented per network type.
	 */
	protected abstract void train();

	/**
	 * Compute the average network error.
	 */
	protected void computeAverageError() {
		double averageOriginalError = 0.0;
		double averageMutantError = 0.0;
		
		//FIXME use ThreadLocalRandom?
		int numTrainingSamples = (int)(trainingIndices.size() * TRAINING_SAMPLE_RATIO);
		Random random = new Random();
		List<Integer> sampledTrainingIndices = new ArrayList<Integer>();
		for(int sample = 0; sample < numTrainingSamples; sample++) {
			int index = random.nextInt(trainingIndices.size());
			sampledTrainingIndices.add(trainingIndices.get(index));
		}
		
		for (Integer trainingIndex : sampledTrainingIndices) {
			if (original.getAverageError() == -1.0) {
				// this may be memoized
				averageOriginalError += original.error(data[trainingIndex], y[trainingIndex]);
			}
			averageMutantError += mutant.error(data[trainingIndex], y[trainingIndex]);
		}
		// calculate original average error
		if (original.getAverageError() == -1.0) {
			averageOriginalError /= trainingIndices.size();
			original.setAverageError(averageOriginalError);
		}
		averageMutantError /= trainingIndices.size();
		mutant.setAverageError(averageMutantError);

		// double improvement = averageOriginalError - averageMutantError;
		// if (improvement > 0.0) {
		// System.out.printf("%.2f ", improvement);
		// }
	}

	public double getError() {
		return error;
	}

	public N getOriginal() {
		return original;
	}

	public N getMutant() {
		return mutant;
	}

}