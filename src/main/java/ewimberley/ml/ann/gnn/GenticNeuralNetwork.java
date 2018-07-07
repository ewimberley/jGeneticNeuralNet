package ewimberley.ml.ann.gnn;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.concurrent.ExecutorService;

import ewimberley.ml.ann.NeuralNetwork;
import ewimberley.ml.ann.NeuralNetworkTrainingConfiguration;
import ewimberley.ml.ann.Neuron;
import ewimberley.ml.ann.gnn.classifier.ClassificationGenticNeuralNetworkWorker;

/**
 * A neural network that evolves as a mechanism for learning.
 * @author ewimberley
 *
 * @param <Y> the output type of the network
 */
public abstract class GenticNeuralNetwork<Y> extends NeuralNetwork<Y> {

	// FIXME make this configurable
	protected static final int NUM_THREADS = Runtime.getRuntime().availableProcessors() * 2;

	protected static void waitForAllWorkers(ExecutorService executor) {
		executor.shutdown();
		while (!executor.isTerminated()) {
			try {
				Thread.sleep(25);
			} catch (InterruptedException e) {
				// do nothing
			}
		}
	}

	protected static PriorityQueue<GenticNeuralNetwork<?>> repopulate(NeuralNetworkTrainingConfiguration config, PriorityQueue<GenticNeuralNetwork<?>> survivors) {
		PriorityQueue<GenticNeuralNetwork<?>> population = new PriorityQueue<GenticNeuralNetwork<?>>(new GenticNeuralNetworkErrorComparator());;
		int numNetworks = 0;
		while (numNetworks < config.getNumNetworksPerGeneration()) {
			GenticNeuralNetwork<?> network = survivors.poll();
			population.add(network);
			numNetworks++;
		}
		return population;
	}

	protected static PriorityQueue<GenticNeuralNetwork<?>> processWorkers(List<GeneticNeuralNetworkWorker<?, ?>> workers) {
		PriorityQueue<GenticNeuralNetwork<?>> survivors = new PriorityQueue<GenticNeuralNetwork<?>>(
				new GenticNeuralNetworkErrorComparator());
		Set<String> survivorIds = new HashSet<String>();
		for (GeneticNeuralNetworkWorker<?, ?> worker : workers) {
			GenticNeuralNetwork<?> mutant = (GenticNeuralNetwork<?>) worker.getMutant();
			GenticNeuralNetwork<?> original = (GenticNeuralNetwork<?>) worker.getOriginal();
			survivors.add(mutant);
			survivorIds.add(mutant.getId());
			if (!survivorIds.contains(original.getId())) {
				survivors.add(original);
				survivorIds.add(original.getId());
			}
		}
		return survivors;
	}

	private double averageError;

	public GenticNeuralNetwork(double[][] data, Y[] y) {
		super(data, y);
		averageError = -1.0;
	}

	/**
	 * Mutate all neurons in the network.
	 */
	public void mutate() {
		generateNewId();
		double learningRateDelta = 0.0;
		if(getRandomDouble() > 0.5) {
			learningRateDelta = this.learningRate * getRandomDouble();
		} else {
			learningRateDelta = -this.learningRate * getRandomDouble();
		}
		this.learningRate += learningRateDelta;  
		for (Map.Entry<String, Neuron> neuronEntry : getNeurons().entrySet()) {
			if (neuronEntry.getValue() instanceof GeneticNeuron) {
				((GeneticNeuron) neuronEntry.getValue()).mutate();
			}
		}
	}

	/**
	 * Compute the error of the current network for a training point.
	 * 
	 * @param inputData
	 *            the input
	 * @param expected
	 *            the expected output
	 * @return the network error
	 */
	public abstract double error(double[] inputData, Y expected);

	/**
	 * Get the average training error for this network.
	 * 
	 * @return the average error across all training samples
	 */
	public double getAverageError() {
		return averageError;
	}

	/**
	 * Set the average training error for this network.
	 * 
	 * @param averageError
	 *            the average error across all training samples
	 */
	public void setAverageError(double averageError) {
		this.averageError = averageError;
	}

}
