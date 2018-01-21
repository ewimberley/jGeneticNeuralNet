package ewimberley.ml.ann.gnn;

import ewimberley.ml.ann.ActivationFunction;
import ewimberley.ml.ann.NeuralNetwork;
import ewimberley.ml.ann.NeuronImpl;

/**
 * Neurons that extend this class return doubles as their activation output
 * 
 * @author ewimberley
 *
 */
public abstract class GeneticNeuron extends NeuronImpl {

	private static final double PROB_MUTATE_BIAS = 0.1;

	private static final double PROB_MUTATE_SYNAPSE_WEIGHT = 0.1;

	/**
	 * Make a deep clone of a continuous neuron.
	 * 
	 * @param network
	 *            the new network the clone belongs to
	 * @param toClone
	 *            the neuron to make a clone of
	 */
	public GeneticNeuron(NeuralNetwork<?> network, NeuronImpl toClone) {
		super(network, toClone);
		this.bias = toClone.getBias();
		this.activationFunction = ActivationFunction.ARCTAN;
	}

	public GeneticNeuron(NeuralNetwork<?> network) {
		super(network);
		this.activationFunction = ActivationFunction.ARCTAN;
	}

	/**
	 * Randomize the entire neuron, used to create brand new neurons.
	 */
	public void scramble() {
		boolean biasNegative = (network.getRandomDouble() > 0.5);
		setBias(network.getRandomDouble() * network.getLearningRate() * (biasNegative ? -1.0 : 1.0));
		for (String nextNeuron : nextWeights.keySet()) {
			boolean weightNegative = (network.getRandomDouble() > 0.5);
			nextWeights.put(nextNeuron,
					(network.getRandomDouble() * network.getLearningRate() * (weightNegative ? -1.0 : 1.0)));
		}
	}

	/**
	 * Slightly randomize the properties of this neuron.
	 */
	public void mutate() {
		double mutateBias = network.getRandomDouble();
		if (mutateBias <= PROB_MUTATE_BIAS) {
			boolean biasNegative = (network.getRandomDouble() > 0.5);
			double deltaBias = network.getRandomDouble() * (biasNegative ? -1.0 : 1.0) * network.getLearningRate();
			setBias(getBias() + deltaBias);
		}
		for (String nextNeuron : nextWeights.keySet()) {
			double mutateSynapseWeight = network.getRandomDouble();
			if (mutateSynapseWeight <= PROB_MUTATE_SYNAPSE_WEIGHT) {
				boolean weightNegative = (network.getRandomDouble() > 0.5);
				double deltaWeight = (network.getRandomDouble() * (weightNegative ? -1.0 : 1.0))
						* network.getLearningRate();
				double newWeight = nextWeights.get(nextNeuron) + deltaWeight;
				nextWeights.put(nextNeuron, newWeight);
			}
		}
	}

}
