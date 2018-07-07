package ewimberley.ml.ann.gnn;

import ewimberley.ml.ann.ActivationFunction;
import ewimberley.ml.ann.HiddenNeuron;
import ewimberley.ml.ann.NeuralNetwork;

public class ContinuousHiddenNeuron extends GeneticNeuron implements HiddenNeuron {

	/*
	 * A default mutation rate in case it isn't set in the config.
	 */
	
	private double probMutateActivationFunction = 0.03;

	public ContinuousHiddenNeuron(NeuralNetwork<?> network, ContinuousHiddenNeuron toClone) {
		super(network, toClone);
		
		double configMutateActivationFunction = ((GeneticNeuralNetworkTrainingConfiguration)network.getConfig()).getProbMutateActivationFunction();
		if(configMutateActivationFunction > 0.0) {
			probMutateActivationFunction = configMutateActivationFunction;
		}
	}

	public ContinuousHiddenNeuron(NeuralNetwork<?> network) {
		super(network);
		
		double configMutateActivationFunction = ((GeneticNeuralNetworkTrainingConfiguration)network.getConfig()).getProbMutateActivationFunction();
		if(configMutateActivationFunction > 0.0) {
			probMutateActivationFunction = configMutateActivationFunction;
		}
	}

	@Override
	public void mutate() {
		super.mutate();
		double mutateActivationFunc = network.getRandomDouble();
		if (mutateActivationFunc <= probMutateActivationFunction) {
			int activationFunc = network.getRandInt(0, 4);
			activationFunction = ActivationFunction.values()[activationFunc];
		}
	}

}
