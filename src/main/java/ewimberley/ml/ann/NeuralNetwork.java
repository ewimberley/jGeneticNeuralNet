package ewimberley.ml.ann;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import ewimberley.ml.Learner;

/**
 * The parent class of all neural networks.
 * 
 * @author ewimberley
 *
 * @param <Y>
 *            the type being predicted
 */
public abstract class NeuralNetwork<Y> extends Learner<Y> {

	protected Map<String, Neuron> neurons;

	protected Set<InputNeuron> inputs;

	protected Map<Integer, InputNeuron> featureToInputMap;

	private List<Set<String>> layers;

	protected Map<OutputNeuron, Y> outputs;

	protected int numHiddenLayers;

	protected int numNeuronsPerLayer;

	private double learningRate;

	private double annealingRate;

	public NeuralNetwork() {
		rand = new Random();
		layers = new ArrayList<Set<String>>();
		inputs = new HashSet<InputNeuron>();
		outputs = new HashMap<OutputNeuron, Y>();
		featureToInputMap = new HashMap<Integer, InputNeuron>();
	}

	protected void createInputLayer() {
		int numInputs = getData()[0].length;
		inputs = new HashSet<InputNeuron>();
		featureToInputMap = new HashMap<Integer, InputNeuron>();
		for (int i = 0; i < numInputs; i++) {
			InputNeuron input = new InputNeuron(this);
			featureToInputMap.put(i, input);
			addInput(input);
		}
	}

	protected void addInput(InputNeuron input) {
		inputs.add(input);
		neurons.put(input.getUuid(), input);
	}

	protected void addOutput(Y y, OutputNeuron output) {
		// FIXME only used for classification?
		outputs.put(output, y);
		neurons.put(output.getUuid(), output);
	}

	/**
	 * Completely randomize every neuron in the network. This is used to create a
	 * new random network.
	 */
	public void scramble() {
		for (Neuron neuron : getNeurons().values()) {
			neuron.scramble();
		}
	}

	public void printNetwork() {
		System.out.println("***********************************");
		System.out.println("Input Layer:");
		for (InputNeuron input : inputs) {
			System.out.println(" " + input.toString());
		}
		for (int i = 1; i < layers.size(); i++) {
			Set<String> layer = layers.get(i);
			System.out.println("Hidden Layer " + i + ":");
			for (String neuronId : layer) {
				Neuron n = neurons.get(neuronId);
				System.out.println(" " + n.toString());
			}
		}
		System.out.println("Output Layer:");
		for (Neuron output : outputs.keySet()) {
			System.out.println(" " + output.toString());
		}
	}

	public Map<String, Neuron> getNeurons() {
		return neurons;
	}

	public Set<InputNeuron> getInputs() {
		return inputs;
	}

	public Map<OutputNeuron, Y> getOutputs() {
		return outputs;
	}

	public int getNumHiddenLayers() {
		return numHiddenLayers;
	}

	public void setNumHiddenLayers(int numHiddenLayers) {
		if (numHiddenLayers < 1) {
			throw new IllegalArgumentException("Number of hidden layers must be at least 1.");
		}
		this.numHiddenLayers = numHiddenLayers;
	}

	public int getNumNeuronsPerLayer() {
		return numNeuronsPerLayer;
	}

	public void setNumNeuronsPerLayer(int numNeuronsPerLayer) {
		if (numNeuronsPerLayer < 1) {
			throw new IllegalArgumentException("Number of neurons per layer must be at least 1.");
		}
		this.numNeuronsPerLayer = numNeuronsPerLayer;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public double getAnnealingRate() {
		return annealingRate;
	}

	public void setAnnealingRate(double annealingRate) {
		this.annealingRate = annealingRate;
	}

	public Map<Integer, InputNeuron> getFeatureToInputMap() {
		return featureToInputMap;
	}

	public List<Set<String>> getLayers() {
		return layers;
	}

	public void setLayers(List<Set<String>> layers) {
		this.layers = layers;
	}

}