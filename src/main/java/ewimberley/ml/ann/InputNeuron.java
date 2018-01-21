package ewimberley.ml.ann;

import java.text.DecimalFormat;

/**
 * A neuron that provides input to the network.
 * @author ewimberley
 *
 * @param <H> the type of output from the neuron
 */
public class InputNeuron extends NeuronImpl {

	private double input;
	
	public InputNeuron(NeuralNetwork<?> network, InputNeuron toClone) {
		super(network, toClone);
		this.input = toClone.getInput();
	}
	
	public InputNeuron(NeuralNetwork<?> network) {
		super(network);
	}

	public double activation() {
		return input;
	}

	public double getInput() {
		return input;
	}

	public void setInput(double input) {
		this.input = input;
	}
	
	@Override
	public String toString() {
		DecimalFormat df = new DecimalFormat();
		df.setMaximumFractionDigits(NeuronImpl.NUM_DECIMALS_TO_STRING);
		String output = "Input neuron " + NeuronImpl.truncatedUUID(getUuid()) + " and next [";
		boolean first = true;
		for (String nextId : next) {
			if (!first) {
				output += ", ";
			} else {
				first = false;
			}
			output += truncatedUUID(nextId) + "::" + df.format(nextWeights.get(nextId));
		}
		output += "]\t<-\t" + input;
		return output;
	}

	public void mutate() {
		//do nothing
	}

	public void scramble() {
		//note that input neurons have no bias
		for (String nextNeuron : nextWeights.keySet()) {
			boolean weightNegative = (network.getRandomDouble() > 0.5);
			nextWeights.put(nextNeuron,
					(network.getRandomDouble() * network.getLearningRate() * (weightNegative ? -1.0 : 1.0)));
		}
	}

}
