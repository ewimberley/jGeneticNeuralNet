package ewimberley.ml.gnn;

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

import ewimberley.ml.Classifier;

public class GenticNeuralNetwork extends NeuralNetwork {

	private static final int NUM_THREADS = 100;

	private double averageError;

	public GenticNeuralNetwork(NeuralNetwork toClone) {
		this(toClone.getData(), toClone.getClassLabels());
		for (Map.Entry<String, Neuron> neuronEntry : toClone.getNeurons().entrySet()) {
			String id = neuronEntry.getKey();
			Neuron neuron = neuronEntry.getValue();
			if (neuron instanceof InputNeuron) {
				InputNeuron clone = new InputNeuron(this, (InputNeuron) neuron);
				addInput(clone);
			} else if (neuron instanceof OutputNeuron) {
				OutputNeuron clone = new OutputNeuron(this, (OutputNeuron) neuron);
				String classLabel = toClone.getOutputs().get(neuron);
				addOutput(classLabel, clone);
			} else {
				Neuron clone = new HiddenNeuron(this, (HiddenNeuron) neuron);
				neurons.put(id, clone);
			}
		}
		for (Map.Entry<Integer, InputNeuron> inputMappingEntry : toClone.getFeatureToInputMap().entrySet()) {
			featureToInputMap.put(inputMappingEntry.getKey(),
					(InputNeuron) neurons.get(inputMappingEntry.getValue().getUuid()));
		}
		this.layers = toClone.layers;
		setLearningRate(toClone.getLearningRate() * (1.0 - toClone.getAnnealingRate()));
	}

	public GenticNeuralNetwork(double[][] data, String[] classLabels) {
		super();
		setLearningRate(0.10); // reasonable default
		setAnnealingRate(0.000001);
		this.setData(data);
		this.setClassLabels(classLabels);
		this.neurons = new HashMap<String, Neuron>();
		numHiddenLayers = 1; // reasonable default
		numNeuronsPerLayer = 5; // reasonable default
		averageError = -1.0;
	}

	public static Classifier train(double[][] data, String[] classLabels, int numNetworksPerGeneration,
			int numGenerations, int numHiddenLayers, int numNeuronsPerLayer, double learningRate) {
		Set<String> uniqueClassLabels = calculateUniqueClassLabels(classLabels);
		Map<String, Integer> classLabelConfusionMatrixIndices = new HashMap<String, Integer>();
		String[] confusionMatrixIndicesToClassLabel = new String[uniqueClassLabels.size()];
		int onConfusionMatrixIndex = 0;
		for (String uniqueClassLabel : uniqueClassLabels) {
			classLabelConfusionMatrixIndices.put(uniqueClassLabel, onConfusionMatrixIndex);
			confusionMatrixIndicesToClassLabel[onConfusionMatrixIndex] = uniqueClassLabel;
			onConfusionMatrixIndex++;
		}
		// first dimension is expected, second dimension is predicted
		int[][] confusionMatrix = new int[uniqueClassLabels.size()][uniqueClassLabels.size()];

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
		NeuralNetwork bestNetwork = null;
		PriorityQueue<GenticNeuralNetwork> population = new PriorityQueue<GenticNeuralNetwork>(
				new NeuralNetworkErrorComparator());
		for (int i = 0; i < numNetworksPerGeneration; i++) {
			GenticNeuralNetwork network = new GenticNeuralNetwork(data, classLabels);
			network.setLearningRate(learningRate);
			network.setNumHiddenLayers(numHiddenLayers);
			network.setNumNeuronsPerLayer(numNeuronsPerLayer);
			network.init();
			network.scramble();
			population.add(network);
			// network.printNetwork();
		}
		for (int gen = 1; gen <= numGenerations; gen++) {
			System.out.println("On generation " + gen + " Population size: " + population.size());
			PriorityQueue<GenticNeuralNetwork> survivors = new PriorityQueue<GenticNeuralNetwork>(
					new NeuralNetworkErrorComparator());
			ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
			List<NeuralNetworkWorker> workers = new ArrayList<NeuralNetworkWorker>();
			for (int onNetwork = 0; onNetwork < numNetworksPerGeneration; onNetwork++) {
				GenticNeuralNetwork network = population.poll();
				// System.out.println("Evaluating network w/ avg error: " +
				// network.getAverageError());
				NeuralNetworkWorker worker = new NeuralNetworkWorker(network, data, classLabels, trainingIndices);
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
			for (NeuralNetworkWorker worker : workers) {
				GenticNeuralNetwork mutant = worker.getMutant();
				double averageMutantError = mutant.getAverageError();
				GenticNeuralNetwork original = worker.getOriginal();
				double averageOriginalError = original.getAverageError();
				if (averageMutantError != -1.0 && averageMutantError < averageOriginalError) {
					// System.out.println("Mutant is better by " + (averageOriginalError -
					// averageMutantError));
					survivors.add(mutant);
					if (averageMutantError < bestAverageError) {
						bestNetwork = (NeuralNetwork) mutant;
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
		bestNetwork.test(data, classLabels, confusionMatrix, testingIndices, classLabelConfusionMatrixIndices);
		printConfusionMatix(confusionMatrixIndicesToClassLabel, confusionMatrix);
		return bestNetwork;
	}

	public void init() {
		createInputLayer();

		// create hidden layers with random links
		Set<Neuron> previousLayer = null;
		Set<Neuron> currentLayer = new HashSet<Neuron>();
		Set<String> currentLayerIds = new HashSet<String>();
		for (InputNeuron input : inputs) {
			currentLayer.add(input);
			currentLayerIds.add(input.getUuid());
		}
		for (int i = 0; i < numHiddenLayers; i++) {
			previousLayer = currentLayer;
			layers.add(currentLayerIds);
			currentLayer = new HashSet<Neuron>();
			currentLayerIds = new HashSet<String>();
			for (int j = 0; j < numNeuronsPerLayer; j++) {
				Neuron n = createNewRandomNeuron();
				currentLayer.add(n);
				currentLayerIds.add(n.getUuid());
			}
			for (Neuron prev : previousLayer) {
				for (Neuron next : currentLayer) {
					prev.addNext(next, 0.0);
				}
			}
		}

		previousLayer = currentLayer;
		layers.add(currentLayerIds);
		currentLayer = new HashSet<Neuron>();
		currentLayerIds = new HashSet<String>();

		// create output layer
		createOutputLayer(previousLayer, currentLayer, currentLayerIds);
	}

	private void createOutputLayer(Set<Neuron> previousLayer, Set<Neuron> currentLayer, Set<String> currentLayerIds) {
		uniqueClassLabels = calculateUniqueClassLabels(getClassLabels());
		outputs = new HashMap<OutputNeuron, String>();
		String[] lableStrings = uniqueClassLabels.toArray(new String[] {});
		for (int i = 0; i < uniqueClassLabels.size(); i++) {
			OutputNeuron output = new OutputNeuron(this);
			addOutput(lableStrings[i], output);
			currentLayer.add(output);
			currentLayerIds.add(output.getUuid());
		}
		for (Neuron prev : previousLayer) {
			for (Neuron next : currentLayer) {
				prev.addNext(next, 0.0);
			}
		}
	}

	private static HashSet<String> calculateUniqueClassLabels(String[] classLabels) {
		HashSet<String> uniqueClassLabels = new HashSet<String>();
		for (String classLabel : classLabels) {
			uniqueClassLabels.add(classLabel);
		}
		return uniqueClassLabels;
	}

	public void mutate() {
		for (Map.Entry<String, Neuron> neuronEntry : getNeurons().entrySet()) {
			neuronEntry.getValue().mutate();
		}
	}

	public double getAverageError() {
		return averageError;
	}

	public void setAverageError(double averageError) {
		this.averageError = averageError;
	}

}
