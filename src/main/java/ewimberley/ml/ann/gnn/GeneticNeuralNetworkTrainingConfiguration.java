package ewimberley.ml.ann.gnn;

import ewimberley.ml.ann.NeuralNetworkTrainingConfiguration;

/**
 * The configuration object for a genetic neural network.
 * 
 * @author ewimberley
 *
 */
public class GeneticNeuralNetworkTrainingConfiguration extends NeuralNetworkTrainingConfiguration {

	private int numNetworksPerGeneration;
	private int numGenerations;
	private double annealingRate;
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
