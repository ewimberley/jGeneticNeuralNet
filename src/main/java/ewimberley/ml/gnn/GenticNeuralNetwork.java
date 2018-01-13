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
import ewimberley.ml.ConfusionMatrix;

public class GenticNeuralNetwork<H> extends NeuralNetwork<H> {

	private static final int NUM_THREADS = 100;

	private double averageError;

	public GenticNeuralNetwork(NeuralNetwork<H> toClone) {
		this(toClone.getData(), toClone.getClassLabels());
		for (Map.Entry<String, Neuron<H>> neuronEntry : toClone.getNeurons().entrySet()) {
			String id = neuronEntry.getKey();
			Neuron<H> neuron = neuronEntry.getValue();
			if (neuron instanceof InputNeuron) {
				InputNeuron<H> clone = new InputNeuron<H>(this, (InputNeuron<H>) neuron);
				addInput(clone);
			} else if (neuron instanceof OutputNeuron) {
				OutputNeuron<H> clone = new OutputNeuron<H>(this, (OutputNeuron<H>) neuron);
				String classLabel = toClone.getOutputs().get(neuron);
				addOutput(classLabel, clone);
			} else {
				Neuron<H> clone = new HiddenNeuron<H>(this, (HiddenNeuron<H>) neuron);
				neurons.put(id, clone);
			}
		}
		for (Map.Entry<Integer, InputNeuron<H>> inputMappingEntry : toClone.getFeatureToInputMap().entrySet()) {
			featureToInputMap.put(inputMappingEntry.getKey(),
					(InputNeuron<H>) neurons.get(inputMappingEntry.getValue().getUuid()));
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
		this.neurons = new HashMap<String, Neuron<H>>();
		numHiddenLayers = 1; // reasonable default
		numNeuronsPerLayer = 5; // reasonable default
		averageError = -1.0;
	}

	public static Classifier train(double[][] data, String[] classLabels, int numNetworksPerGeneration,
			int numGenerations, int numHiddenLayers, int numNeuronsPerLayer, double learningRate) {
		ConfusionMatrix cf = new ConfusionMatrix(classLabels);

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
		NeuralNetwork<Double> bestNetwork = null;
		PriorityQueue<GenticNeuralNetwork<Double>> population = new PriorityQueue<GenticNeuralNetwork<Double>>(
				new NeuralNetworkErrorComparator());
		for (int i = 0; i < numNetworksPerGeneration; i++) {
			GenticNeuralNetwork<Double> network = new GenticNeuralNetwork<Double>(data, classLabels);
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
			PriorityQueue<GenticNeuralNetwork<Double>> survivors = new PriorityQueue<GenticNeuralNetwork<Double>>(
					new NeuralNetworkErrorComparator());
			ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
			List<NeuralNetworkWorker<Double>> workers = new ArrayList<NeuralNetworkWorker<Double>>();
			for (int onNetwork = 0; onNetwork < numNetworksPerGeneration; onNetwork++) {
				GenticNeuralNetwork<Double> network = population.poll();
				// System.out.println("Evaluating network w/ avg error: " +
				// network.getAverageError());
				NeuralNetworkWorker<Double> worker = new NeuralNetworkWorker<Double>(network, data, classLabels, trainingIndices);
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
			for (NeuralNetworkWorker<Double> worker : workers) {
				GenticNeuralNetwork<Double> mutant = worker.getMutant();
				double averageMutantError = mutant.getAverageError();
				GenticNeuralNetwork<Double> original = worker.getOriginal();
				double averageOriginalError = original.getAverageError();
				if (averageMutantError != -1.0 && averageMutantError < averageOriginalError) {
					// System.out.println("Mutant is better by " + (averageOriginalError -
					// averageMutantError));
					survivors.add(mutant);
					if (averageMutantError < bestAverageError) {
						bestNetwork = (NeuralNetwork<Double>) mutant;
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
		bestNetwork.test(data, classLabels, cf, testingIndices);
		cf.printConfusionMatix();
		return bestNetwork;
	}

	public void init() {
		createInputLayer();

		// create hidden layers with random links
		Set<Neuron<H>> previousLayer = null;
		Set<Neuron<H>> currentLayer = new HashSet<Neuron<H>>();
		Set<String> currentLayerIds = new HashSet<String>();
		for (InputNeuron<H> input : inputs) {
			currentLayer.add(input);
			currentLayerIds.add(input.getUuid());
		}
		for (int i = 0; i < numHiddenLayers; i++) {
			previousLayer = currentLayer;
			layers.add(currentLayerIds);
			currentLayer = new HashSet<Neuron<H>>();
			currentLayerIds = new HashSet<String>();
			for (int j = 0; j < numNeuronsPerLayer; j++) {
				Neuron<H> n = createNewRandomNeuron();
				currentLayer.add(n);
				currentLayerIds.add(n.getUuid());
			}
			for (Neuron<H> prev : previousLayer) {
				for (Neuron<H> next : currentLayer) {
					prev.addNext(next, 0.0);
				}
			}
		}

		previousLayer = currentLayer;
		layers.add(currentLayerIds);
		currentLayer = new HashSet<Neuron<H>>();
		currentLayerIds = new HashSet<String>();

		// create output layer
		createOutputLayer(previousLayer, currentLayer, currentLayerIds);
	}

	private void createOutputLayer(Set<Neuron<H>> previousLayer, Set<Neuron<H>> currentLayer, Set<String> currentLayerIds) {
		uniqueClassLabels = calculateUniqueClassLabels(getClassLabels());
		outputs = new HashMap<OutputNeuron<H>, String>();
		String[] lableStrings = uniqueClassLabels.toArray(new String[] {});
		for (int i = 0; i < uniqueClassLabels.size(); i++) {
			OutputNeuron<H> output = new OutputNeuron<H>(this);
			addOutput(lableStrings[i], output);
			currentLayer.add(output);
			currentLayerIds.add(output.getUuid());
		}
		for (Neuron<H> prev : previousLayer) {
			for (Neuron<H> next : currentLayer) {
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
		for (Map.Entry<String, Neuron<H>> neuronEntry : getNeurons().entrySet()) {
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
