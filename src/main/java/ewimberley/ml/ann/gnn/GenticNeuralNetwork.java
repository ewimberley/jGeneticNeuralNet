package ewimberley.ml.ann.gnn;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

import ewimberley.ml.ann.NeuralNetwork;
import ewimberley.ml.ann.Neuron;

public class GenticNeuralNetwork<H> extends NeuralNetwork<H> {

	protected static final int NUM_THREADS = 100;

	private double averageError;

	public GenticNeuralNetwork(double[][] data, String[] classLabels) {
		super();
		setLearningRate(0.10); // reasonable default
		setAnnealingRate(0.000001);
		this.setData(data);
		this.setClassLabels(classLabels);
		this.neurons = new HashMap<String, Neuron<H>>();
		numHiddenLayers = 1; // reasonable default
		numNeuronsPerLayer = 5; // reasonable default
		averageError = -1.0;
	}

	protected static HashSet<String> calculateUniqueClassLabels(String[] classLabels) {
		HashSet<String> uniqueClassLabels = new HashSet<String>();
		for (String classLabel : classLabels) {
			uniqueClassLabels.add(classLabel);
		}
		return uniqueClassLabels;
	}

	public void mutate() {
		for (Map.Entry<String, Neuron<H>> neuronEntry : getNeurons().entrySet()) {
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
