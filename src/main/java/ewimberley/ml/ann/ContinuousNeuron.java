package ewimberley.ml.ann;

import java.text.DecimalFormat;

/**
 * Neurons that extend this class return doubles as their activation output
 * 
 * @author ewimberley
 *
 */
public abstract class ContinuousNeuron extends NeuronImpl<Double> {

	private static final double PROB_MUTATE_BIAS = 0.1;
	
	private static final double PROB_MUTATE_SYNAPSE_WEIGHT = 0.1;

	protected double bias;

	protected double memoizedActivation;

	protected ActivationFunction activationFunction;

	/**
	 * Make a deep clone of a continuous neuron.
	 * 
	 * @param network
	 *            the new network the clone belongs to
	 * @param toClone
	 *            the neuron to make a clone of
	 */
	public ContinuousNeuron(NeuralNetwork<Double> network, ContinuousNeuron toClone) {
		super(network, toClone);
		this.bias = toClone.getBias();
		this.activationFunction = ActivationFunction.ARCTAN;
	}

	public ContinuousNeuron(NeuralNetwork<Double> network) {
		super(network);
		this.activationFunction = ActivationFunction.ARCTAN;
	}

	/**
	 * An activation function based on sin(x)
	 * 
	 * @param in
	 *            the sum of inputs and the bias
	 * @return a number between 0 and 1
	 */
	public double sinusoidal(double in) {
		// y=(sin(x*pi-pi/2)+1)/2
		return ((Math.sin(in * Math.PI - Math.PI / 2) + 1) / 2.0);
	}

	/**
	 * An activation function based on arctan(x)
	 * 
	 * @param in
	 *            the sum of inputs and the bias
	 * @return a number between 0 and 1
	 */
	public double arctan(double in) {
		// y=arctan(x)/Pi+0.5
		return Math.atan(in) / Math.PI + 0.5;
	}

	/**
	 * An activation function based on the step function.
	 * 
	 * @param in
	 *            the sum of inputs and the bias
	 * @return a number between 0 and 1
	 */
	public double step(double in) {
		if (in >= 0) {
			return 1.0;
		} else {
			return 0.0;
		}
	}

	public Double activation() {
		if (!memoized) {
			double output = 0.0;
			for (String prevNeuron : prev) {
				Neuron<Double> prevNeuronObj = network.getNeurons().get(prevNeuron);
				double prevOutput = prevNeuronObj.activation();
				double prevWeight = prevNeuronObj.getNextWeights().get(this.getUuid());
				output += prevOutput * prevWeight;
			}
			output += bias;
			if (activationFunction == ActivationFunction.ARCTAN) {
				memoizedActivation = arctan(output);
			} else if (activationFunction == ActivationFunction.SIN) {
				memoizedActivation = sinusoidal(output);
			} else if (activationFunction == ActivationFunction.STEP) {
				memoizedActivation = step(output);
			} else if (activationFunction == ActivationFunction.LINEAR) {
				memoizedActivation = output;
			}
			memoized = true;
		}
		return memoizedActivation;
	}

	public void scramble() {
		boolean biasNegative = (network.getRandomDouble() > 0.5);
		setBias(network.getRandomDouble() * network.getLearningRate() * (biasNegative ? -1.0 : 1.0));
		for (String nextNeuron : nextWeights.keySet()) {
			boolean weightNegative = (network.getRandomDouble() > 0.5);
			nextWeights.put(nextNeuron,
					(network.getRandomDouble() * network.getLearningRate() * (weightNegative ? -1.0 : 1.0)));
		}
	}

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

	@Override
	public String toString() {
		DecimalFormat df = new DecimalFormat();
		df.setMaximumFractionDigits(NUM_DECIMALS_TO_STRING);
		String output = "Neuron " + truncatedUUID(uuid) + " with bias " + df.format(getBias()) + " and next [";
		boolean first = true;
		for (String nextId : next) {
			if (!first) {
				output += ", ";
			} else {
				first = false;
			}
			output += truncatedUUID(nextId) + "::" + df.format(nextWeights.get(nextId));
		}
		output += "]\t->\t" + memoizedActivation;
		return output;
	}

	public double getBias() {
		return bias;
	}

	public void setBias(double bias) {
		this.bias = bias;
	}

	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}

	public void setActivationFunction(ActivationFunction activationFunction) {
		this.activationFunction = activationFunction;
	}

}
