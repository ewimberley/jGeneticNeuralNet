package ewimberley.ml.ann.gnn;

import java.text.DecimalFormat;

import ewimberley.ml.ann.ActivationFunction;
import ewimberley.ml.ann.NeuralNetwork;
import ewimberley.ml.ann.NeuronImpl;
import ewimberley.ml.ann.OutputNeuron;

/**
 * An output neuron with floating point activation function output.
 * 
 * @author ewimberley
 *
 */
public class ContinuousOutputNeuron extends GeneticNeuron implements OutputNeuron {

	//XXX only allow activation functions with range 0.0 to 1.0
	
	public ContinuousOutputNeuron(NeuralNetwork<?> network, ContinuousOutputNeuron toClone) {
		super(network, toClone);
	}

	public ContinuousOutputNeuron(NeuralNetwork<?> network) {
		super(network);
	}
	
//	@Override
//	protected double activation(double in) {
//		double out = 0.0;
//		if (activationFunction == ActivationFunction.ARCTAN) {
//			out = arctan(in);
//		} else if (activationFunction == ActivationFunction.SIN) {
//			out = sinusoidal(in);
//		} else {
//			//range must be between 0 and 1
//			out = super.activation(in);
//			if(out > 1.0) {
//				return 1.0;
//			} else if(out < 0.0) {
//				return 0.0;
//			} else {
//				return out;
//			}
//		}
//		return out;
//	}

	@Override
	public String toString() {
		DecimalFormat df = new DecimalFormat();
		df.setMaximumFractionDigits(NeuronImpl.NUM_DECIMALS_TO_STRING);
		return "Output neuron " + NeuronImpl.truncatedUUID(getUuid()) + " associted with "
				+ network.getOutputs().get(this) + " with bias " + df.format(getBias()) + "\t->\t" + memoizedActivation;
	}

}
