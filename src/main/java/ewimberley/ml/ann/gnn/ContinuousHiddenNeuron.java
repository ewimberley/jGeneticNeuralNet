package ewimberley.ml.ann.gnn;

import ewimberley.ml.ann.ActivationFunction;
import ewimberley.ml.ann.HiddenNeuron;
import ewimberley.ml.ann.NeuralNetwork;

public class ContinuousHiddenNeuron extends GeneticNeuron implements HiddenNeuron {

	private static final double PROB_MUTATE_ACTIVATION_FUNC = 0.03;

	public ContinuousHiddenNeuron(NeuralNetwork<?> network, ContinuousHiddenNeuron toClone) {
		super(network, toClone);
	}

	public ContinuousHiddenNeuron(NeuralNetwork<?> network) {
		super(network);
	}

	@Override
	public void mutate() {
		super.mutate();
		double mutateActivationFunc = network.getRandomDouble();
		if (mutateActivationFunc <= PROB_MUTATE_ACTIVATION_FUNC) {
			int activationFunc = network.getRandInt(0, 3);
			activationFunction = ActivationFunction.values()[activationFunc];
		}
	}

}
