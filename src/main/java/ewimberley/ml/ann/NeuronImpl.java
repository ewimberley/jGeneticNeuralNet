package ewimberley.ml.ann;

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
public abstract class NeuronImpl implements Neuron {

	public static final int NUM_DECIMALS_TO_STRING = 3;

	public static final int NUM_ID_CHARS_TO_STRING = 5;

	protected NeuralNetwork<?> network;

	protected String uuid;

	protected Set<String> next;

	//FIXME push this down to a lower class?
	protected Map<String, Double> nextWeights;

	protected Set<String> prev;

	protected boolean memoized;

	protected double bias;

	protected double memoizedActivation;

	protected ActivationFunction activationFunction;

	/**
	 * Make a deep clone of a neuron.
	 * 
	 * @param network
	 *            the new network the clone belongs to
	 * @param toClone
	 *            the neuron to make a clone of
	 */
	public NeuronImpl(NeuralNetwork<?> network, NeuronImpl toClone) {
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

	public NeuronImpl(NeuralNetwork<?> network) {
		next = new HashSet<String>();
		prev = new HashSet<String>();
		nextWeights = new HashMap<String, Double>();
		this.network = network;
		uuid = java.util.UUID.randomUUID().toString();
	}

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

	/**
	 * An activation function based on sin(x)
	 * 
	 * @param in
	 *            the sum of inputs and the bias
	 * @return a number between 0 and 1
	 */
	public double sinusoidal(double in) {
		// y=(sin(x*pi-pi/2)+1)/2
		return ((Math.sin(in * Math.PI - Math.PI / 2) + 1) / 2.0);
	}

	/**
	 * An activation function based on arctan(x)
	 * 
	 * @param in
	 *            the sum of inputs and the bias
	 * @return a number between 0 and 1
	 */
	public double arctan(double in) {
		// y=arctan(x)/Pi+0.5
		return Math.atan(in) / Math.PI + 0.5;
	}

	/**
	 * An activation function based on the step function.
	 * 
	 * @param in
	 *            the sum of inputs and the bias
	 * @return a number between 0 and 1
	 */
	public double step(double in) {
		if (in >= 0) {
			return 1.0;
		} else {
			return 0.0;
		}
	}
	
	/**
	 * An activation function based on a parabola.
	 * 
	 * @param in
	 *            the sum of inputs and the bias
	 * @return the number squared
	 */
	public double parabolic(double in) {
		return in * in;
	}


	/**
	 * Run the activation function for this neuron.
	 * 
	 * @param in
	 *            the summed input
	 * @return the activation result of the activation function for this neuron
	 */
	protected double activation(double in) {
		double out = 0.0;
		if (activationFunction == ActivationFunction.ARCTAN) {
			out = arctan(in);
		} else if (activationFunction == ActivationFunction.SIN) {
			out = sinusoidal(in);
		} else if (activationFunction == ActivationFunction.STEP) {
			out = step(in);
		} else if (activationFunction == ActivationFunction.LINEAR) {
			out = in;
		} else if (activationFunction == ActivationFunction.PARABOLIC) {
			out = parabolic(in);
		}
		return out;
	}

	/**
	 * Get the output of this neuron.
	 */
	public double activation() {
		if (!memoized) {
			double inputSum = 0.0;
			for (String prevNeuron : prev) {
				Neuron prevNeuronObj = network.getNeurons().get(prevNeuron);
				double prevOutput = prevNeuronObj.activation();
				double prevWeight = prevNeuronObj.getNextWeights().get(this.getUuid());
				inputSum += prevOutput * prevWeight;
			}
			inputSum += bias;
			memoizedActivation = activation(inputSum);
			memoized = true;
		}
		return memoizedActivation;
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

	public double getBias() {
		return bias;
	}

	public void setBias(double bias) {
		this.bias = bias;
	}

	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}

	public void setActivationFunction(ActivationFunction activationFunction) {
		this.activationFunction = activationFunction;
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
