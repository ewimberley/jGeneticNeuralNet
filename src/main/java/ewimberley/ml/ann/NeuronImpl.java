package ewimberley.ml.ann;

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
public abstract class NeuronImpl<H> implements Neuron<H> {

	public static final int NUM_DECIMALS_TO_STRING = 3;

	public static final int NUM_ID_CHARS_TO_STRING = 5;

	protected NeuralNetwork<H,?> network;

	protected String uuid;

	protected Set<String> next;

	protected Map<String, Double> nextWeights;

	protected Set<String> prev;

	protected boolean memoized;

	/**
	 * Make a deep clone of a neuron.
	 * 
	 * @param network
	 *            the new network the clone belongs to
	 * @param toClone
	 *            the neuron to make a clone of
	 */
	public NeuronImpl(NeuralNetwork<H,?> network, NeuronImpl<H> toClone) {
		this(network);
		this.uuid = toClone.getUuid();
		for (String nextNeuron : toClone.getNext()) {
			next.add(nextNeuron);
			nextWeights.put(nextNeuron, toClone.getNextWeights().get(nextNeuron));
		}
		for (String prevNeuron : toClone.getPrev()) {
			prev.add(prevNeuron);
		}
	}

	public NeuronImpl(NeuralNetwork<H,?> network) {
		next = new HashSet<String>();
		prev = new HashSet<String>();
		nextWeights = new HashMap<String, Double>();
		this.network = network;
		uuid = java.util.UUID.randomUUID().toString();
	}

	public Set<String> getNext() {
		return next;
	}

	public void addNext(Neuron<H> next, double weight) {
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

	public void addPrev(String prev) {
		this.prev.add(prev);
	}

	public String getUuid() {
		return uuid;
	}
	
	public void resetMemoization() {
		memoized = false;
	}

	public Map<String, Double> getNextWeights() {
		return nextWeights;
	}

	public static String truncatedUUID(String uuid) {
		return uuid.substring(uuid.length() - NUM_ID_CHARS_TO_STRING, uuid.length());
	}

	@Override
	public int hashCode() {
		return Objects.hash(uuid);
	}

	//TODO write a toString() for this that makes sense and call it from child classes
//	@Override
//	public String toString() {
//		DecimalFormat df = new DecimalFormat();
//		df.setMaximumFractionDigits(NUM_DECIMALS_TO_STRING);
//		String output = "Neuron " + truncatedUUID(uuid) + " with bias " + df.format(getBias()) + " and next [";
//		boolean first = true;
//		for (String nextId : next) {
//			if (!first) {
//				output += ", ";
//			} else {
//				first = false;
//			}
//			output += truncatedUUID(nextId) + "::" + df.format(nextWeights.get(nextId));
//		}
//		output += "]\t->\t" + memoizedActivation;
//		return output;
//	}

}
