package ewimberley.ml.ann.gnn;

import java.util.Map;

import ewimberley.ml.ann.NeuralNetwork;
import ewimberley.ml.ann.Neuron;

public abstract class GenticNeuralNetwork<Y> extends NeuralNetwork<Y> {

	// FIXME make this configurable
	protected static final int NUM_THREADS = 100;

	private double averageError;

	public GenticNeuralNetwork(double[][] data, Y[] y) {
		super(data, y);
		setLearningRate(0.10); // reasonable default
		setAnnealingRate(0.000001);
		numHiddenLayers = 1; // reasonable default
		numNeuronsPerLayer = 5; // reasonable default
		averageError = -1.0;
	}

	/**
	 * Mutate all neurons in the network.
	 */
	public void mutate() {
		for (Map.Entry<String, Neuron> neuronEntry : getNeurons().entrySet()) {
			if (neuronEntry.getValue() instanceof GeneticNeuron) {
				((GeneticNeuron) neuronEntry.getValue()).mutate();
			}
		}
	}

	/**
	 * Compute the error of the current network for a training point.
	 * 
	 * @param inputData
	 *            the input
	 * @param expected
	 *            the expected output
	 * @return the network error
	 */
	public abstract double error(double[] inputData, Y expected);

	/**
	 * Get the average training error for this network.
	 * 
	 * @return the average error across all training samples
	 */
	public double getAverageError() {
		return averageError;
	}

	/**
	 * Set the average training error for this network.
	 * 
	 * @param averageError
	 *            the average error across all training samples
	 */
	public void setAverageError(double averageError) {
		this.averageError = averageError;
	}

}
