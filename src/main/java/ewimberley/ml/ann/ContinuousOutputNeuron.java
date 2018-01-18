package ewimberley.ml.ann;

import java.text.DecimalFormat;

/**
 * An output neuron with floating point activation function output.
 * 
 * @author ewimberley
 *
 */
public class ContinuousOutputNeuron extends ContinuousNeuron implements OutputNeuron<Double> {

	//XXX only allow activation functions with range 0.0 to 1.0
	
	public ContinuousOutputNeuron(NeuralNetwork<Double, ?> network, ContinuousOutputNeuron toClone) {
		super(network, toClone);
	}

	public ContinuousOutputNeuron(NeuralNetwork<Double, ?> network) {
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
