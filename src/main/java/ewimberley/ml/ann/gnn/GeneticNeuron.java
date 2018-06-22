package ewimberley.ml.ann.gnn;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import ewimberley.ml.ann.ActivationFunction;
import ewimberley.ml.ann.NeuralNetwork;
import ewimberley.ml.ann.Neuron;
import ewimberley.ml.ann.NeuronImpl;

/**
 * Neurons that extend this class return doubles as their activation output
 * 
 * @author ewimberley
 *
 */
public abstract class GeneticNeuron extends NeuronImpl {

	private static final double HALF = 0.5;

	/*
	 * These are default values if they aren't set in the config. 
	 */
	
	private double probMutateEdges = 0.01;

	private double probMutateBias = 0.02;

	private double probMutateWeights = 0.05;

	/**
	 * Make a deep clone of a continuous neuron.
	 * 
	 * @param network
	 *            the new network the clone belongs to
	 * @param toClone
	 *            the neuron to make a clone of
	 */
	public GeneticNeuron(NeuralNetwork<?> network, NeuronImpl toClone) {
		super(network, toClone);
		this.bias = toClone.getBias();
		this.activationFunction = toClone.getActivationFunction();
		
		double configMutateEdges = network.getConfig().getProbMutateEdges();
		if(configMutateEdges > 0.0) {
			probMutateEdges = configMutateEdges;
		}
		
		double configMutateBias = network.getConfig().getProbMutateBias();
		if(configMutateBias > 0.0) {
			probMutateBias = configMutateBias;
		}
		
		double configMutateWeights = network.getConfig().getProbMutateWeights();
		if(configMutateWeights > 0.0) {
			probMutateWeights = configMutateWeights;
		}
	}

	public GeneticNeuron(NeuralNetwork<?> network) {
		super(network);
		this.activationFunction = ActivationFunction.ARCTAN;
	}

	/**
	 * Randomize the entire neuron, used to create brand new neurons.
	 */
	public void scramble() {
		boolean biasNegative = (network.getRandomDouble() > 0.5);
		setBias(network.getRandomDouble() * network.getLearningRate() * (biasNegative ? -1.0 : 1.0));
		for (String nextNeuron : nextWeights.keySet()) {
			boolean weightNegative = (network.getRandomDouble() > 0.5);
			nextWeights.put(nextNeuron,
					(network.getRandomDouble() * network.getLearningRate() * (weightNegative ? -1.0 : 1.0)));
		}
	}

	/**
	 * Slightly randomize the properties of this neuron.
	 */
	public void mutate() {
		double mutateBias = network.getRandomDouble();
		if (mutateBias <= probMutateBias) {
			boolean biasNegative = (network.getRandomDouble() > HALF);
			double deltaBias = network.getRandomDouble() * (biasNegative ? -1.0 : 1.0) * network.getLearningRate();
			setBias(getBias() + deltaBias);
		}
		for (String nextNeuron : nextWeights.keySet()) {
			double mutateSynapseWeight = network.getRandomDouble();
			if (mutateSynapseWeight <= probMutateWeights) {
				boolean weightNegative = (network.getRandomDouble() > HALF);
				double deltaWeight = (network.getRandomDouble() * (weightNegative ? -1.0 : 1.0))
						* network.getLearningRate();
				double newWeight = nextWeights.get(nextNeuron) + deltaWeight;
				nextWeights.put(nextNeuron, newWeight);
			}
		}
		mutateStructure();
	}

	private void mutateStructure() {
		double mutateEdges = network.getRandomDouble();
		if (mutateEdges <= probMutateEdges) {
			boolean addNeuron = (network.getRandomDouble() > HALF);
			if (addNeuron) {
				String nextNeuronId = getRandomNeuronInLaterLayer();
				if (nextNeuronId != null) {
					boolean weightNegative = (network.getRandomDouble() > HALF);
					nextWeights.put(nextNeuronId,
							(network.getRandomDouble() * network.getLearningRate() * (weightNegative ? -1.0 : 1.0)));
					Neuron nextNeuron = network.getNeurons().get(nextNeuronId);
					nextNeuron.getPrev().add(this.getUuid());
					next.add(nextNeuronId);
				}
			} else {
				List<String> deleteNeuronList = new ArrayList<String>();
				deleteNeuronList.addAll(nextWeights.keySet());
				if (!deleteNeuronList.isEmpty()) {
					String toDelete = deleteNeuronList.get(network.getRandInt(0, deleteNeuronList.size() - 1));
					nextWeights.remove(toDelete);
					network.getNeurons().get(toDelete).getPrev().remove(this.getUuid());
					next.remove(toDelete);
				}
			}
		}
	}

	/**
	 * Get a random neuron ID from a layer after the layer of this neuron.
	 * 
	 * @return a UUID of a neuron
	 */
	private String getRandomNeuronInLaterLayer() {
		Set<String> possibleNextNeurons = new HashSet<String>();
		boolean possibleNextLayer = false;
		for (Set<String> layer : network.getLayers()) {
			if (possibleNextLayer) {
				for (String neuron : layer) {
					if (!nextWeights.containsKey(neuron)) {
						possibleNextNeurons.add(neuron);
					}
				}
			}
			if (layer.contains(this.getUuid())) {
				possibleNextLayer = true;
			}
		}
		List<String> nextNeuronList = new ArrayList<String>();
		nextNeuronList.addAll(possibleNextNeurons);
		if (nextNeuronList.isEmpty()) {
			return null;
		}
		String nextNeuron = nextNeuronList.get(network.getRandInt(0, nextNeuronList.size() - 1));
		return nextNeuron;
	}

}
