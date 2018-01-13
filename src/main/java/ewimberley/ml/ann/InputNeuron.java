package ewimberley.ml.ann;

import java.text.DecimalFormat;

public class InputNeuron<H> extends NeuronImpl<H> {

	private H input;
	
	public InputNeuron(NeuralNetwork<H> network, InputNeuron<H> toClone) {
		super(network, toClone);
		this.input = toClone.getInput();
	}
	
	public InputNeuron(NeuralNetwork<H> network) {
		super(network);
	}

	public H activation() {
		return input;
	}

	public H getInput() {
		return input;
	}

	public void setInput(H input) {
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
