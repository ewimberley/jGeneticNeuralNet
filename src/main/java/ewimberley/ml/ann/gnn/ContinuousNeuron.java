package ewimberley.ml.ann.gnn;

import java.text.DecimalFormat;

import ewimberley.ml.ann.NeuralNetwork;
import ewimberley.ml.ann.NeuronImpl;

/**
 * Neurons that extend this class return doubles as their activation output
 * 
 * @author ewimberley
 *
 */
public abstract class ContinuousNeuron extends NeuronImpl<Double> {

	protected double bias;

	protected double memoizedActivation;

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
	}

	public ContinuousNeuron(NeuralNetwork<Double> network) {
		super(network);
	}

	public double sinusoidal(double in) {
		// y=(sin(x*pi-pi/2)+1)/2
		return ((Math.sin(in * Math.PI - Math.PI / 2) + 1) / 2.0);
	}

	public double arctan(double in) {
		// y=arctan(x)/Pi+0.5
		return Math.atan(in) / Math.PI + 0.5;
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
		boolean biasNegative = (network.getRandomDouble() > 0.5);
		double deltaBias = network.getRandomDouble() * (biasNegative ? -1.0 : 1.0) * network.getLearningRate();
		setBias(getBias() + deltaBias);
		for (String nextNeuron : nextWeights.keySet()) {
			boolean weightNegative = (network.getRandomDouble() > 0.5);
			double deltaWeight = (network.getRandomDouble() * (weightNegative ? -1.0 : 1.0))
					* network.getLearningRate();
			double newWeight = nextWeights.get(nextNeuron) + deltaWeight;
			nextWeights.put(nextNeuron, newWeight);
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

}
