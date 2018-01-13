package ewimberley.ml.ann;

public class ContinuousHiddenNeuron extends ContinuousNeuron implements HiddenNeuron<Double> {

	private static final double PROB_MUTATE_ACTIVATION_FUNC = 0.03;

	public ContinuousHiddenNeuron(NeuralNetwork<Double> network, ContinuousHiddenNeuron toClone) {
		super(network, toClone);
	}

	public ContinuousHiddenNeuron(NeuralNetwork<Double> network) {
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
