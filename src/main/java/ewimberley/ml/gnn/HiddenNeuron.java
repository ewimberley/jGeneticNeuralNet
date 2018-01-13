package ewimberley.ml.gnn;

public abstract class HiddenNeuron<H> extends Neuron<H> {

	public HiddenNeuron(NeuralNetwork<H> network, HiddenNeuron<H> toClone) {
		super(network, toClone);
	}
	
	public HiddenNeuron(NeuralNetwork<H> network) {
		super(network);
	}

}
