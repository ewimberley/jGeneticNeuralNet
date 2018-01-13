package ewimberley.ml.ann.gnn.regression;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ewimberley.ml.ConfusionMatrix;
import ewimberley.ml.Learner;
import ewimberley.ml.ann.ContinuousHiddenNeuron;
import ewimberley.ml.ann.ContinuousOutputNeuron;
import ewimberley.ml.ann.InputNeuron;
import ewimberley.ml.ann.NeuralNetwork;
import ewimberley.ml.ann.Neuron;
import ewimberley.ml.ann.NeuronImpl;
import ewimberley.ml.ann.OutputNeuron;
import ewimberley.ml.ann.gnn.GenticNeuralNetwork;
import ewimberley.ml.ann.gnn.GenticNeuralNetworkErrorComparator;

public class RegressionGenticNeuralNetwork extends GenticNeuralNetwork<Double, Double> {
	
	private OutputNeuron<Double> output;

	public RegressionGenticNeuralNetwork(double[][] data, Double[] y) {
		super(data, y);
	}

	public RegressionGenticNeuralNetwork(NeuralNetwork<Double, Double> toClone) {
		super(toClone.getData(), toClone.getY());
		for (Map.Entry<String, Neuron<Double>> neuronEntry : toClone.getNeurons().entrySet()) {
			String id = neuronEntry.getKey();
			Neuron<Double> neuron = neuronEntry.getValue();
			if (neuron instanceof InputNeuron) {
				InputNeuron<Double> clone = new InputNeuron<Double>(this, (InputNeuron<Double>) neuron);
				addInput(clone);
			} else if (neuron instanceof OutputNeuron) {
				OutputNeuron<Double> clone = new ContinuousOutputNeuron(this, (ContinuousOutputNeuron) neuron);
				output = clone;
			} else {
				NeuronImpl<Double> clone = new ContinuousHiddenNeuron(this, (ContinuousHiddenNeuron) neuron);
				neurons.put(id, clone);
			}
		}
		for (Map.Entry<Integer, InputNeuron<Double>> inputMappingEntry : toClone.getFeatureToInputMap().entrySet()) {
			featureToInputMap.put(inputMappingEntry.getKey(),
					(InputNeuron<Double>) neurons.get(inputMappingEntry.getValue().getUuid()));
		}
		this.setLayers(toClone.getLayers());
		setLearningRate(toClone.getLearningRate() * (1.0 - toClone.getAnnealingRate()));
	}

	public static Learner<Double> train(double[][] data, double[] yPrim, int numNetworksPerGeneration,
			int numGenerations, int numHiddenLayers, int numNeuronsPerLayer, double learningRate) {
		Double[] y = new Double[yPrim.length];
		for (int i = 0; i < yPrim.length; i++) {
			y[i] = yPrim[i];
		}
		// FIXME implement 10-fold cross validation
		// currently random training/testing set
		List<Integer> dataIndices = new ArrayList<Integer>();
		for (int i = 0; i < data.length; i++) {
			dataIndices.add(i);
		}
		Collections.shuffle(dataIndices);
		int numTesting = data.length / 5;
		List<Integer> trainingIndices = new ArrayList<Integer>();
		List<Integer> testingIndices = new ArrayList<Integer>();
		for (int i = 0; i < data.length; i++) {
			if (i > numTesting) {
				trainingIndices.add(dataIndices.get(i));
			} else {
				testingIndices.add(dataIndices.get(i));
			}
		}

		double bestAverageError = Double.MAX_VALUE;
		NeuralNetwork<Double, Double> bestNetwork = null;
		PriorityQueue<RegressionGenticNeuralNetwork> population = new PriorityQueue<RegressionGenticNeuralNetwork>(
				new GenticNeuralNetworkErrorComparator<Double>());
		for (int i = 0; i < numNetworksPerGeneration; i++) {
			RegressionGenticNeuralNetwork network = new RegressionGenticNeuralNetwork(data, y);
			network.setLearningRate(learningRate);
			network.setNumHiddenLayers(numHiddenLayers);
			network.setNumNeuronsPerLayer(numNeuronsPerLayer);
			network.init();
			network.scramble();
			population.add(network);
			bestNetwork = network;
			// network.printNetwork();
		}
		for (int gen = 1; gen <= numGenerations; gen++) {
			System.out.println("On generation " + gen + " Population size: " + population.size());
			PriorityQueue<RegressionGenticNeuralNetwork> survivors = new PriorityQueue<RegressionGenticNeuralNetwork>(
					new GenticNeuralNetworkErrorComparator<Double>());
			ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
			List<RegressionGenticNeuralNetworkWorker> workers = new ArrayList<RegressionGenticNeuralNetworkWorker>();
			for (int onNetwork = 0; onNetwork < numNetworksPerGeneration; onNetwork++) {
				RegressionGenticNeuralNetwork network = population.poll();
				// System.out.println("Evaluating network w/ avg error: " +
				// network.getAverageError());
				RegressionGenticNeuralNetworkWorker worker = new RegressionGenticNeuralNetworkWorker(network, data, y,
						trainingIndices);
				workers.add(worker);
				executor.execute(worker);

			}
			executor.shutdown();
			while (!executor.isTerminated()) {
				try {
					Thread.sleep(25);
				} catch (InterruptedException e) {
					// do nothing
				}
			}
			for (RegressionGenticNeuralNetworkWorker worker : workers) {
				RegressionGenticNeuralNetwork mutant = (RegressionGenticNeuralNetwork) worker.getMutant();
				double averageMutantError = mutant.getAverageError();
				RegressionGenticNeuralNetwork original = (RegressionGenticNeuralNetwork) worker.getOriginal();
				double averageOriginalError = original.getAverageError();
				if (averageMutantError != -1.0 && averageMutantError < averageOriginalError) {
					// System.out.println("Mutant is better by " + (averageOriginalError -
					// averageMutantError));
					survivors.add(mutant);
					if (averageMutantError < bestAverageError) {
						bestNetwork = (NeuralNetwork<Double, Double>) mutant;
						bestAverageError = averageMutantError;
					}
				} else {
					survivors.add(original);
					if (averageOriginalError != -1.0 && averageOriginalError < bestAverageError) {
						bestNetwork = original;
						bestAverageError = averageOriginalError;
					}
				}

			}
			population = survivors;
			// bestNetwork.printNetwork();
			System.out.println("Best network had average error: " + bestAverageError);
		}
		// System.out.println("Best network had average error: " + bestAverageError);
		// bestNetwork.printNetwork();
		((RegressionGenticNeuralNetwork) bestNetwork).test(data, y, testingIndices);
		return bestNetwork;
	}

	public void init() {
		createInputLayer();

		// create hidden layers with random links
		Set<Neuron<Double>> previousLayer = null;
		Set<Neuron<Double>> currentLayer = new HashSet<Neuron<Double>>();
		Set<String> currentLayerIds = new HashSet<String>();
		for (InputNeuron<Double> input : inputs) {
			currentLayer.add(input);
			currentLayerIds.add(input.getUuid());
		}
		for (int i = 0; i < numHiddenLayers; i++) {
			previousLayer = currentLayer;
			getLayers().add(currentLayerIds);
			currentLayer = new HashSet<Neuron<Double>>();
			currentLayerIds = new HashSet<String>();
			for (int j = 0; j < numNeuronsPerLayer; j++) {
				Neuron<Double> n = createNewRandomNeuron();
				currentLayer.add(n);
				currentLayerIds.add(n.getUuid());
			}
			for (Neuron<Double> prev : previousLayer) {
				for (Neuron<Double> next : currentLayer) {
					prev.addNext(next, 0.0);
				}
			}
		}

		previousLayer = currentLayer;
		getLayers().add(currentLayerIds);
		currentLayer = new HashSet<Neuron<Double>>();
		currentLayerIds = new HashSet<String>();

		// create output layer
		createOutputLayer(previousLayer, currentLayer, currentLayerIds);
	}

	protected void createOutputLayer(Set<Neuron<Double>> previousLayer, Set<Neuron<Double>> currentLayer,
			Set<String> currentLayerIds) {
		output = new ContinuousOutputNeuron(this);
		neurons.put(output.getUuid(), output);
		currentLayer.add(output);
		currentLayerIds.add(output.getUuid());
		for (Neuron<Double> prev : previousLayer) {
			for (Neuron<Double> next : currentLayer) {
				prev.addNext(next, 0.0);
			}
		}
		getLayers().add(currentLayerIds);
	}

	public void test(double[][] inputData, Double[] expected, List<Integer> testingIndices) {
		for (Integer testingIndex : testingIndices) {
			setupForPredict(inputData[testingIndex]);
			double highestProb = 0.0;
			String highestProbClass = null;
//			for (Neuron<Double> output : outputs.keySet()) {
//				double prob = output.activation();
//				if (prob > highestProb) {
//					highestProb = prob;
//					highestProbClass = outputs.get(output);
//				}
//			}
//			System.out.println("Expected " + expectedClass + ", predicted " + highestProbClass + " with probability "
//					+ highestProb);
		}
	}

	public double error(double[] inputData, Double expected) {
		double totalError = 0.0;
		//FIXME use mean squared error?
		setupForPredict(inputData);
		for (Neuron<Double> output : outputs.keySet()) {
			double prob = output.activation();
			double error = Math.abs(expected - prob);
			totalError += error;
		}
		return totalError;
	}

	public Double predict(double[] inputData) {
		double highestProb = 0.0;
		Neuron<Double> highestProbNeuron = null;
		setupForPredict(inputData);
		for (Neuron<Double> output : outputs.keySet()) {
			double prob = output.activation();
			if (prob > highestProb) {
				highestProbNeuron = output;
				highestProb = prob;
			}
		}
		System.out.println("Predicted class is " + outputs.get(highestProbNeuron) + " with prob value " + highestProb);
		return output.activation();
	}

	protected void setupForPredict(double[] inputData) {
		for (Map.Entry<String, Neuron<Double>> neuronEntry : getNeurons().entrySet()) {
			neuronEntry.getValue().resetMemoization();
		}
		for (int i = 0; i < inputData.length; i++) {
			InputNeuron<Double> in = featureToInputMap.get(i);
			in.setInput(inputData[i]);
		}
	}

	protected Neuron<Double> createNewRandomNeuron() {
		Neuron<Double> n = new ContinuousHiddenNeuron(this);
		neurons.put(n.getUuid(), n);
		return n;
	}

}
