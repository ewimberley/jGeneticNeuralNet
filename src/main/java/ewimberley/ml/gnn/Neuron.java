package ewimberley.ml.gnn;

import java.util.Map;
import java.util.Set;

public interface Neuron<H> {

	H activation();

	Set<String> getNext();

	void addNext(Neuron<H> next, double weight);

	void addNext(String next, double weight);

	Set<String> getPrev();

	String getUuid();

	Map<String, Double> getNextWeights();

	void scramble();

	void resetMemoization();

	void mutate();

	void addPrev(String prev);

}