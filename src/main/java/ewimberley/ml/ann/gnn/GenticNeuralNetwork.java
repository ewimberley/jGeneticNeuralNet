package ewimberley.ml.ann.gnn;

import java.util.HashMap;
import java.util.Map;

import ewimberley.ml.ann.NeuralNetwork;
import ewimberley.ml.ann.Neuron;

public abstract class GenticNeuralNetwork<Y> extends NeuralNetwork<Y> {

	protected static final int NUM_THREADS = 100;

	private double averageError;

	public GenticNeuralNetwork(double[][] data, Y[] y) {
		super();
		setLearningRate(0.10); // reasonable default
		setAnnealingRate(0.000001);
		this.setData(data);
		this.setY(y);
		this.neurons = new HashMap<String, Neuron>();
		numHiddenLayers = 1; // reasonable default
		numNeuronsPerLayer = 5; // reasonable default
		averageError = -1.0;
	}

	public void mutate() {
		for (Map.Entry<String, Neuron> neuronEntry : getNeurons().entrySet()) {
			neuronEntry.getValue().mutate();
		}
	}

	public double getAverageError() {
		return averageError;
	}

	public void setAverageError(double averageError) {
		this.averageError = averageError;
	}

}
