package ewimberley.ml.ann;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ewimberley.ml.ConfusionMatrix;
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

	private String id;

	protected Map<String, Neuron> neurons;

	protected Set<InputNeuron> inputs;

	protected Map<Integer, InputNeuron> featureToInputMap;

	private List<Set<String>> layers;

	protected Map<OutputNeuron, Y> outputs;

	protected NeuralNetworkTrainingConfiguration config;

	protected double learningRate;

	private ConfusionMatrix confusion;

	public NeuralNetwork(double[][] data, Y[] y) {
		super(data, y);
		generateNewId();
		layers = new ArrayList<Set<String>>();
		inputs = new HashSet<InputNeuron>();
		outputs = new HashMap<OutputNeuron, Y>();
		neurons = new HashMap<String, Neuron>();
		featureToInputMap = new HashMap<Integer, InputNeuron>();
	}

	/**
	 * Create a valid input layer with at least one neuron per input feature.
	 */
	protected void createInputLayer() {
		// XXX nominal variables: create an input neuron for every category w/ 0/1 input
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

	/**
	 * Print the network in a human readable form for debugging purposes.
	 */
	public void printNetwork() {
		System.out.println("***********************************");
		System.out.println("Input Layer:");
		for (InputNeuron input : inputs) {
			System.out.println(" " + input.toString());
		}
		for (int i = 1; i < layers.size() - 1; i++) {
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
		return config.getNumHiddenLayers();
	}

	public int getNumNeuronsPerLayer() {
		return config.getNumNeuronsPerLayer();
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public Map<Integer, InputNeuron> getFeatureToInputMap() {
		return featureToInputMap;
	}

	/**
	 * Get the layers of the neural network as a set of IDs.
	 * 
	 * @return a list of sets where each item is a layer of the network
	 */
	public List<Set<String>> getLayers() {
		return layers;
	}

	public void setLayers(List<Set<String>> layers) {
		this.layers = layers;
	}

	/**
	 * Get the unique id of this network.
	 * 
	 * @return the unique id string.
	 */
	public String getId() {
		return id;
	}

	protected void generateNewId() {
		id = java.util.UUID.randomUUID().toString();
	}

	/**
	 * Get the training configuration.
	 * 
	 * @return the training configuration object
	 */
	public NeuralNetworkTrainingConfiguration getConfig() {
		return config;
	}

	/**
	 * Set the training configuration.
	 * 
	 * @param config
	 *            the training configuration object
	 */
	public void setConfig(NeuralNetworkTrainingConfiguration config) {
		this.config = config;
	}

	/**
	 * Get the testing confusion matrix for this network (if it has been tested).
	 * 
	 * @return the confusion matrix
	 */
	public ConfusionMatrix getConfusion() {
		return confusion;
	}

	/**
	 * Set the testing confusion matrix for this network.
	 * 
	 * @param confusion
	 *            the testing confusion matrix
	 */
	public void setConfusion(ConfusionMatrix confusion) {
		this.confusion = confusion;
	}

}