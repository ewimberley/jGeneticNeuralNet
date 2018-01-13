package ewimberley.ml.ann;

import java.text.DecimalFormat;

public class ContinuousOutputNeuron extends ContinuousNeuron implements OutputNeuron<Double> {

	public ContinuousOutputNeuron(NeuralNetwork<Double,?> network, ContinuousOutputNeuron toClone) {
		super(network, toClone);
	}

	public ContinuousOutputNeuron(NeuralNetwork<Double,?> network) {
		super(network);
	}

	@Override
	public String toString() {
		DecimalFormat df = new DecimalFormat();
		df.setMaximumFractionDigits(NeuronImpl.NUM_DECIMALS_TO_STRING);
		return "Output neuron " + NeuronImpl.truncatedUUID(getUuid()) + " associted with "
				+ network.getOutputs().get(this) + " with bias " + df.format(getBias()) + "\t->\t" + memoizedActivation;
	}

}
