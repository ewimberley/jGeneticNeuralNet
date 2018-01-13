package ewimberley.ml.gnn;

public abstract class ContinuousNeuron extends NeuronImpl<Double> {

	public ContinuousNeuron(NeuralNetwork<Double> network, ContinuousNeuron toClone) {
		super(network, toClone);
	}
	
	public ContinuousNeuron(NeuralNetwork<Double> network) {
		super(network);
	}

}
