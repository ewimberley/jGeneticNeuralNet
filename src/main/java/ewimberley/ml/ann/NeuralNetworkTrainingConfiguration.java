package ewimberley.ml.ann;

import ewimberley.ml.ModelTrainingConfig;
import ewimberley.ml.ann.visualizer.ANNVisualizer;

/**
 * A training configuration object for all neural networks.
 * 
 * @author ewimberley
 *
 */
public class NeuralNetworkTrainingConfiguration extends ModelTrainingConfig {

	private int numHiddenLayers;
	private int numNeuronsPerLayer;
	private double maxLearningRate;
	private ANNVisualizer visualizer;
	public NeuralNetworkTrainingConfiguration() {
		super();
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

}