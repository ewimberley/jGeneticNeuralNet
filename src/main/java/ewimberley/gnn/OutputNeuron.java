package ewimberley.gnn;

import java.text.DecimalFormat;

public class OutputNeuron extends Neuron {

	public OutputNeuron(NeuralNetwork network, OutputNeuron toClone) {
		super(network, toClone);
	}
	
	public OutputNeuron(NeuralNetwork network) {
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
				Neuron prevNeuronObj = network.getNeurons().get(prevNeuron);
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
