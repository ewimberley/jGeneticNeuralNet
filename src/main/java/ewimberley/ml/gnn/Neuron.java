package ewimberley.ml.gnn;

import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * The parent class of all neurons.
 * 
 * @author ewimberley
 *
 */
public abstract class Neuron<H> implements Cloneable {

	public static final int NUM_DECIMALS_TO_STRING = 3;

	public static final int NUM_ID_CHARS_TO_STRING = 5;

	protected NeuralNetwork<H> network;

	protected String uuid;

	protected Set<String> next;

	protected Map<String, Double> nextWeights;

	protected Set<String> prev;

	protected double bias;

	protected boolean memoized;

	protected double memoizedActivation;

	public Neuron(NeuralNetwork network, Neuron<H> toClone) {
		this(network);
		this.uuid = toClone.getUuid();
		for (String nextNeuron : toClone.getNext()) {
			next.add(nextNeuron);
			nextWeights.put(nextNeuron, toClone.getNextWeights().get(nextNeuron));
		}
		for (String prevNeuron : toClone.getPrev()) {
			prev.add(prevNeuron);
		}
		this.bias = toClone.getBias();
	}

	public Neuron(NeuralNetwork network) {
		next = new HashSet<String>();
		prev = new HashSet<String>();
		nextWeights = new HashMap<String, Double>();
		this.network = network;
		uuid = java.util.UUID.randomUUID().toString();
	}

	public double sinusoidal(double in) {
		// y=(sin(x*pi-pi/2)+1)/2
		return ((Math.sin(in * Math.PI - Math.PI / 2) + 1) / 2.0);
	}

	public double arctan(double in) {
		// y=arctan(x)/Pi+0.5
		return Math.atan(in) / Math.PI + 0.5;
	}

	//public abstract H activation();
	public abstract double activation();

	public Set<String> getNext() {
		return next;
	}

	public void addNext(Neuron next, double weight) {
		addNext(next.getUuid(), weight);
	}

	public void addNext(String next, double weight) {
		this.next.add(next);
		this.getNextWeights().put(next, weight);
		this.network.getNeurons().get(next).addPrev(this.getUuid());
	}

	public Set<String> getPrev() {
		return prev;
	}

	private void addPrev(String prev) {
		this.prev.add(prev);
	}

	public double getBias() {
		return bias;
	}

	public void setBias(double bias) {
		this.bias = bias;
	}

	public String getUuid() {
		return uuid;
	}

	@Override
	public int hashCode() {
		return Objects.hash(uuid);
	}

	public Map<String, Double> getNextWeights() {
		return nextWeights;
	}

	public void scramble() {
		boolean biasNegative = (network.getRandomDouble() > 0.5);
		setBias(network.getRandomDouble() * network.getLearningRate() * (biasNegative ? -1.0 : 1.0));
		// setBias(0.0);
		for (String nextNeuron : nextWeights.keySet()) {
			// nextWeights.put(nextNeuron, 1.0);
			boolean weightNegative = (network.getRandomDouble() > 0.5);
			nextWeights.put(nextNeuron,
					(network.getRandomDouble() * network.getLearningRate() * (weightNegative ? -1.0 : 1.0)));
		}
	}

	public void mutate() {
		boolean biasNegative = (network.getRandomDouble() > 0.5);
		double deltaBias = network.getRandomDouble() * (biasNegative ? -1.0 : 1.0) * network.getLearningRate();
		setBias(getBias() + deltaBias);
		for (String nextNeuron : nextWeights.keySet()) {
			boolean weightNegative = (network.getRandomDouble() > 0.5);
			double deltaWeight = (network.getRandomDouble() * (weightNegative ? -1.0 : 1.0))
					* network.getLearningRate();
			double newWeight = nextWeights.get(nextNeuron) + deltaWeight;
			nextWeights.put(nextNeuron, newWeight);
		}
	}

	public static String truncatedUUID(String uuid) {
		return uuid.substring(uuid.length() - NUM_ID_CHARS_TO_STRING, uuid.length());
	}

	@Override
	public String toString() {
		DecimalFormat df = new DecimalFormat();
		df.setMaximumFractionDigits(NUM_DECIMALS_TO_STRING);
		String output = "Neuron " + truncatedUUID(uuid) + " with bias " + df.format(getBias()) + " and next [";
		boolean first = true;
		for (String nextId : next) {
			if (!first) {
				output += ", ";
			} else {
				first = false;
			}
			output += truncatedUUID(nextId) + "::" + df.format(nextWeights.get(nextId));
		}
		output += "]\t->\t" + memoizedActivation;
		return output;
	}

	public void resetMemoization() {
		memoized = false;
	}

}
