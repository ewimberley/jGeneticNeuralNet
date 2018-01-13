package ewimberley.ml.gnn;

public class ContinuousHiddenNeuron extends ContinuousNeuron implements HiddenNeuron<Double> {

	public ContinuousHiddenNeuron(NeuralNetwork<Double> network, ContinuousHiddenNeuron toClone) {
		super(network, toClone);
	}

	public ContinuousHiddenNeuron(NeuralNetwork<Double> network) {
		super(network);
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
