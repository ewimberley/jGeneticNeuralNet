package ewimberley.ml.gnn;

import java.text.DecimalFormat;

public class HiddenNeuron<H> extends Neuron<H> {

	public HiddenNeuron(NeuralNetwork<H> network, HiddenNeuron<H> toClone) {
		super(network, toClone);
	}
	
	public HiddenNeuron(NeuralNetwork<H> network) {
		super(network);
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
