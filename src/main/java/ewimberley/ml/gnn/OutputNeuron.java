package ewimberley.ml.gnn;

import java.text.DecimalFormat;

public class OutputNeuron<H> extends Neuron<H> {

	public OutputNeuron(NeuralNetwork<H> network, OutputNeuron<H> toClone) {
		super(network, toClone);
	}
	
	public OutputNeuron(NeuralNetwork<H> network) {
		super(network);
	}
	
	@Override
	public String toString() {
		DecimalFormat df = new DecimalFormat();
		df.setMaximumFractionDigits(Neuron.NUM_DECIMALS_TO_STRING);
		return "Output neuron " + Neuron.truncatedUUID(getUuid()) + " associted with " + network.getOutputs().get(this) + " with bias " + df.format(getBias()) + "\t->\t" + memoizedActivation;
	}
	
	public double activation() {
		if (!memoized) {
			double output = 0.0;
			for (String prevNeuron : prev) {
				Neuron<H> prevNeuronObj = network.getNeurons().get(prevNeuron);
				double prevOutput = prevNeuronObj.activation();
				double prevWeight = prevNeuronObj.getNextWeights().get(this.getUuid());
				output += prevOutput * prevWeight;
			}
			output += bias;
			memoizedActivation = arctan(output);
			memoized = true;
		}
		return memoizedActivation;
	}

}
