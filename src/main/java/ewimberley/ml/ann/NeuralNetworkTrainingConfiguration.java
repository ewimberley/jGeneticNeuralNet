package ewimberley.ml.ann;

import ewimberley.ml.ann.visualizer.ANNVisualizer;

public class NeuralNetworkTrainingConfiguration {

	private int numNetworksPerGeneration;
	private int numGenerations;
	private int numHiddenLayers;
	private int numNeuronsPerLayer;
	private double maxLearningRate;
	private double annealingRate;
	private ANNVisualizer visualizer;
	private int maxThreads;
	private double probMutateEdges;
	private double probMutateBias;
	private double probMutateWeights;
	private double probMutateActivationFunction;

	public int getNumNetworksPerGeneration() {
		return numNetworksPerGeneration;
	}

	public void setNumNetworksPerGeneration(int numNetworksPerGeneration) {
		this.numNetworksPerGeneration = numNetworksPerGeneration;
	}

	public int getNumGenerations() {
		return numGenerations;
	}

	public void setNumGenerations(int numGenerations) {
		this.numGenerations = numGenerations;
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

	public double getMaxLearningRate() {
		return maxLearningRate;
	}

	public void setMaxLearningRate(double learningRate) {
		this.maxLearningRate = learningRate;
	}

	public ANNVisualizer getVisualizer() {
		return visualizer;
	}

	public void setVisualizer(ANNVisualizer visualizer) {
		this.visualizer = visualizer;
	}

	public int getMaxThreads() {
		return maxThreads;
	}

	public void setMaxThreads(int maxThreads) {
		if (maxThreads < 1) {
			throw new IllegalArgumentException("Number of threads must be at least 1.");
		}
		this.maxThreads = maxThreads;
	}

	public double getProbMutateEdges() {
		return probMutateEdges;
	}

	public void setProbMutateEdges(double probMutateEdges) {
		this.probMutateEdges = probMutateEdges;
	}

	public double getProbMutateBias() {
		return probMutateBias;
	}

	public void setProbMutateBias(double probMutateBias) {
		this.probMutateBias = probMutateBias;
	}

	public double getProbMutateWeights() {
		return probMutateWeights;
	}

	public void setProbMutateWeights(double probMutateWeights) {
		this.probMutateWeights = probMutateWeights;
	}

	public double getProbMutateActivationFunction() {
		return probMutateActivationFunction;
	}

	public void setProbMutateActivationFunction(double probMutateActivationFunction) {
		this.probMutateActivationFunction = probMutateActivationFunction;
	}

	public double getAnnealingRate() {
		return annealingRate;
	}

	public void setAnnealingRate(double annealingRate) {
		this.annealingRate = annealingRate;
	}

}
