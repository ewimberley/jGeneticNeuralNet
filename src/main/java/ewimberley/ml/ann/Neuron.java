package ewimberley.ml.ann;

import java.util.Map;
import java.util.Set;

/**
 * Classes that implement this interface can be used as neurons in neural
 * networks.
 * 
 * @author ewimberley
 *
 * @param <H>
 *            the return type of the activation function
 */
public interface Neuron<H> {

	/**
	 * Calculate the output of the neuron (recursively).
	 * 
	 * @return the value of this neuron based on network inputs
	 */
	H activation();

	/**
	 * Get a set of neuron ids that this neuron passes its activation output to.
	 * 
	 * @return a set of neuron ids
	 */
	Set<String> getNext();

	/**
	 * Add a neuron to the list of neurons this neuron passes its activation output
	 * to.
	 * 
	 * @param next
	 *            the neuron to add
	 * @param weight
	 *            the weight of this neuron's output on the next neurons activation
	 *            function
	 */
	void addNext(Neuron<H> next, double weight);

	/**
	 * Add a neuron to the list of neurons this neuron passes its activation output
	 * to.
	 * 
	 * @param next
	 *            the id of the next neuron
	 * @param weight
	 *            the weight of this neuron's output on the next neurons activation
	 *            function
	 */
	void addNext(String next, double weight);

	/**
	 * Get a set of neuron ids that pass their activation functions to this neuron.
	 * 
	 * @return a set of neuron ids
	 */
	Set<String> getPrev();

	/**
	 * Get the id of this neuron.
	 * 
	 * @return a unique id string
	 */
	String getUuid();

	/**
	 * Get a map from next neuron ids to weights.
	 * 
	 * @return a map from neuron ids to weights
	 */
	Map<String, Double> getNextWeights();

	/**
	 * Randomly initialize this neuron.
	 */
	void scramble();

	/**
	 * Call this after providing a new set of inputs to the network before calling
	 * predict.
	 */
	void resetMemoization();

	//XXX push this down to a genetic neural network specific neuron class/interface.
	/**
	 * Randomly alter this neuron by a small amount.
	 */
	void mutate();

	/**
	 * Add a neuron to the list of neurons that pass their activation functions to
	 * this neuron.
	 * 
	 * @param prev
	 *            the id of the previous neuron
	 */
	void addPrev(String prev);

}