package ewimberley.ml.gnn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import ewimberley.ml.Classifier;
import ewimberley.ml.ConfusionMatrix;

public abstract class NeuralNetwork<H> extends Classifier {

	protected Map<String, Neuron<H>> neurons;
	protected Set<InputNeuron<H>> inputs;
	protected Map<Integer, InputNeuron<H>> featureToInputMap;
	protected List<Set<String>> layers;
	protected Map<OutputNeuron<H>, String> outputs;
	protected int numHiddenLayers;
	protected int numNeuronsPerLayer;
	private double learningRate;
	private double annealingRate;

	public NeuralNetwork() {
		rand = new Random();
		layers = new ArrayList<Set<String>>();
		inputs = new HashSet<InputNeuron<H>>();
		outputs = new HashMap<OutputNeuron<H>, String>();
		featureToInputMap = new HashMap<Integer, InputNeuron<H>>();
	}

	public void test(double[][] inputData, String[] expected, ConfusionMatrix cf, List<Integer> testingIndices) {
		for (Integer testingIndex : testingIndices) {
			setupForPredict(inputData[testingIndex]);
			double highestProb = 0.0;
			String highestProbClass = null;
			for (Neuron<H> output : outputs.keySet()) {
				double prob = output.activation();
				if(prob > highestProb) {
					highestProb = prob;
					highestProbClass = outputs.get(output);
				}
			}
			String expectedClass = expected[testingIndex];
			System.out.println("Expected " + expectedClass + ", predicted " + highestProbClass + " with probability " + highestProb);
			cf.getConfusionMatrix()[cf.getClassLabelConfusionMatrixIndices().get(expectedClass)][cf.getClassLabelConfusionMatrixIndices().get(highestProbClass)]++;
		}
	}

	public double error(double[] inputData, String expected) {
		double totalError = 0.0;
		setupForPredict(inputData);
		for (Neuron<H> output : outputs.keySet()) {
			double prob = output.activation();
			double error = 0.0;
			if (outputs.get(output).equals(expected)) {
				error = Math.abs(1.0 - prob);
			} else {
				error = Math.abs(0.0 - prob);
			}
			totalError += error;
		}
		return totalError;
	}

	// FIXME figure out how to return class and probability
	public String predict(double[] inputData) {
		double highestProb = 0.0;
		Neuron<H> highestProbNeuron = null;
		setupForPredict(inputData);
		for (Neuron<H> output : outputs.keySet()) {
			double prob = output.activation();
			if (prob > highestProb) {
				highestProbNeuron = output;
				highestProb = prob;
			}
		}
		System.out.println("Predicted class is " + outputs.get(highestProbNeuron) + " with prob value " + highestProb);
		return outputs.get(highestProbNeuron);
	}

	private void setupForPredict(double[] inputData) {
		for (Map.Entry<String, Neuron<H>> neuronEntry : getNeurons().entrySet()) {
			neuronEntry.getValue().resetMemoization();
		}
		for (int i = 0; i < inputData.length; i++) {
			InputNeuron<H> in = featureToInputMap.get(i);
			in.setInput(inputData[i]);
		}
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

	protected Neuron<H> createNewRandomNeuron() {
		Neuron<H> n = new HiddenNeuron<H>(this);
		neurons.put(n.getUuid(), n);
		return n;
	}

	protected void addOutput(String classLabel, OutputNeuron<H> output) {
		outputs.put(output, classLabel);
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

	public Map<OutputNeuron<H>, String> getOutputs() {
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

}