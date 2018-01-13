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
 * @author ewimberley
 *
 * @param <H> the output of activation functions
 * @param <Y> the type being predicted
 */
public abstract class NeuralNetwork<H, Y> extends Learner<Y> {

	protected Map<String, Neuron<H>> neurons;
	
	protected Set<InputNeuron<H>> inputs;
	
	protected Map<Integer, InputNeuron<H>> featureToInputMap;
	
	private List<Set<String>> layers;
	
	protected Map<OutputNeuron<H>, Y> outputs;
	
	protected int numHiddenLayers;
	
	protected int numNeuronsPerLayer;
	
	private double learningRate;
	
	private double annealingRate;

	public NeuralNetwork() {
		rand = new Random();
		layers = new ArrayList<Set<String>>();
		inputs = new HashSet<InputNeuron<H>>();
		outputs = new HashMap<OutputNeuron<H>, Y>();
		featureToInputMap = new HashMap<Integer, InputNeuron<H>>();
	}

	protected void createInputLayer() {
		int numInputs = getData()[0].length;
		inputs = new HashSet<InputNeuron<H>>();
		featureToInputMap = new HashMap<Integer, InputNeuron<H>>();
		for (int i = 0; i < numInputs; i++) {
			InputNeuron<H> input = new InputNeuron<H>(this);
			featureToInputMap.put(i, input);
			addInput(input);
		}
	}

	protected void addInput(InputNeuron<H> input) {
		inputs.add(input);
		neurons.put(input.getUuid(), input);
	}
	
	protected void addOutput(Y y, OutputNeuron<H> output) {
		//FIXME only used for classification?
		outputs.put(output, y);
		neurons.put(output.getUuid(), output);
	}

	public void scramble() {
		for (Neuron<H> neuron : getNeurons().values()) {
			neuron.scramble();
		}
	}

	public void printNetwork() {
		System.out.println("***********************************");
		System.out.println("Input Layer:");
		for (InputNeuron<H> input : inputs) {
			System.out.println(" " + input.toString());
		}
		for (int i = 1; i < layers.size(); i++) {
			Set<String> layer = layers.get(i);
			System.out.println("Hidden Layer " + i + ":");
			for (String neuronId : layer) {
				Neuron<H> n = neurons.get(neuronId);
				System.out.println(" " + n.toString());
			}
		}
		System.out.println("Output Layer:");
		for (Neuron<H> output : outputs.keySet()) {
			System.out.println(" " + output.toString());
		}
	}

	public Map<String, Neuron<H>> getNeurons() {
		return neurons;
	}

	public Set<InputNeuron<H>> getInputs() {
		return inputs;
	}

	public Map<OutputNeuron<H>, Y> getOutputs() {
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

	public Map<Integer, InputNeuron<H>> getFeatureToInputMap() {
		return featureToInputMap;
	}

	public List<Set<String>> getLayers() {
		return layers;
	}

	public void setLayers(List<Set<String>> layers) {
		this.layers = layers;
	}

}