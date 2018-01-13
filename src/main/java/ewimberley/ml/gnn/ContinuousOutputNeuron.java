package ewimberley.ml.gnn;

import java.text.DecimalFormat;

public class ContinuousOutputNeuron extends ContinuousNeuron implements OutputNeuron<Double> {

	public ContinuousOutputNeuron(NeuralNetwork<Double> network, ContinuousOutputNeuron toClone) {
		super(network, toClone);
	}

	public ContinuousOutputNeuron(NeuralNetwork<Double> network) {
		super(network);
	}

	@Override
	public String toString() {
		DecimalFormat df = new DecimalFormat();
		df.setMaximumFractionDigits(NeuronImpl.NUM_DECIMALS_TO_STRING);
		return "Output neuron " + NeuronImpl.truncatedUUID(getUuid()) + " associted with "
				+ network.getOutputs().get(this) + " with bias " + df.format(getBias()) + "\t->\t" + memoizedActivation;
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
			memoizedActivation = arctan(output);
			memoized = true;
		}
		return memoizedActivation;
	}

}
