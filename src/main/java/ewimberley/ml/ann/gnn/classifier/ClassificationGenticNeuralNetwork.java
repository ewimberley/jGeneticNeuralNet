package ewimberley.ml.ann.gnn.classifier;

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
import ewimberley.ml.ann.InputNeuron;
import ewimberley.ml.ann.NeuralNetwork;
import ewimberley.ml.ann.NeuralNetworkTrainingConfiguration;
import ewimberley.ml.ann.Neuron;
import ewimberley.ml.ann.NeuronImpl;
import ewimberley.ml.ann.OutputNeuron;
import ewimberley.ml.ann.gnn.ContinuousHiddenNeuron;
import ewimberley.ml.ann.gnn.ContinuousOutputNeuron;
import ewimberley.ml.ann.gnn.GeneticNeuralNetworkWorker;
import ewimberley.ml.ann.gnn.GenticNeuralNetwork;
import ewimberley.ml.ann.gnn.GenticNeuralNetworkErrorComparator;
import ewimberley.ml.ann.gnn.regression.RegressionGenticNeuralNetwork;
import ewimberley.ml.ann.visualizer.ANNVisualizer;

public class ClassificationGenticNeuralNetwork extends GenticNeuralNetwork<String> {

	private static final int GENERATIONAL_DEBUG_INTERVAL = 10;

	public ClassificationGenticNeuralNetwork(double[][] data, String[] classLabels) {
		super(data, classLabels);
	}

	public ClassificationGenticNeuralNetwork(NeuralNetwork<String> toClone) {
		super(toClone.getData(), toClone.getY());
		this.setConfig(toClone.getConfig());
		for (Map.Entry<String, Neuron> neuronEntry : toClone.getNeurons().entrySet()) {
			String id = neuronEntry.getKey();
			Neuron neuron = neuronEntry.getValue();
			if (neuron instanceof InputNeuron) {
				InputNeuron clone = new InputNeuron(this, (InputNeuron) neuron);
				addInput(clone);
			} else if (neuron instanceof OutputNeuron) {
				OutputNeuron clone = new ContinuousOutputNeuron(this, (ContinuousOutputNeuron) neuron);
				String classLabel = toClone.getOutputs().get(neuron);
				addOutput(classLabel, clone);
			} else {
				NeuronImpl clone = new ContinuousHiddenNeuron(this, (ContinuousHiddenNeuron) neuron);
				neurons.put(id, clone);
			}
		}
		for (Map.Entry<Integer, InputNeuron> inputMappingEntry : toClone.getFeatureToInputMap().entrySet()) {
			featureToInputMap.put(inputMappingEntry.getKey(),
					(InputNeuron) neurons.get(inputMappingEntry.getValue().getUuid()));
		}
		this.setLayers(toClone.getLayers());
		setLearningRate(toClone.getLearningRate() * (1.0 - toClone.getAnnealingRate()));
	}

	public static GenticNeuralNetwork<String> train(double[][] data, String[] classLabels,
			NeuralNetworkTrainingConfiguration config) {
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

		GenticNeuralNetwork<?> bestNetwork = null;
		PriorityQueue<GenticNeuralNetwork<?>> population = new PriorityQueue<GenticNeuralNetwork<?>>(
				new GenticNeuralNetworkErrorComparator());
		for (int i = 0; i < config.getNumNetworksPerGeneration(); i++) {
			ClassificationGenticNeuralNetwork network = new ClassificationGenticNeuralNetwork(data, classLabels);
			double randLearningRate = network.getRandomDouble() * config.getMaxLearningRate();
			network.setLearningRate(randLearningRate);
			network.setConfig(config);
			network.init(config.getNumHiddenLayers(), config.getNumNeuronsPerLayer());
			network.scramble();
			population.add(network);
			bestNetwork = network;
		}
		for (int gen = 1; gen <= config.getNumGenerations(); gen++) {
			if ((gen % GENERATIONAL_DEBUG_INTERVAL) == 0) {
				System.out.print("On generation " + gen + " Population size: " + population.size());
			}
			ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
			List<GeneticNeuralNetworkWorker<?, ?>> workers = new ArrayList<GeneticNeuralNetworkWorker<?, ?>>();
			int numNetworks = 0;
			double numChildrenPerNetwork = config.getNumNetworksPerGeneration() / 100;
			if (numChildrenPerNetwork < 30.0) {
				numChildrenPerNetwork = 30.0;
			}
			while (numNetworks < config.getNumNetworksPerGeneration()) {
				if (numChildrenPerNetwork > 1) {
					numChildrenPerNetwork -= 1.0;
				}
				int intNumChildrenPerNetwork = (int) Math.ceil(numChildrenPerNetwork);
				ClassificationGenticNeuralNetwork network = (ClassificationGenticNeuralNetwork) population.poll();
				for (int onChild = 0; onChild < intNumChildrenPerNetwork; onChild++) {
					ClassificationGenticNeuralNetworkWorker worker = new ClassificationGenticNeuralNetworkWorker(
							network, data, classLabels, trainingIndices);
					workers.add(worker);
					executor.execute(worker);
					numNetworks++;
				}
			}
			waitForAllWorkers(executor);
			PriorityQueue<GenticNeuralNetwork<?>> workersOutput = processWorkers(workers);
			bestNetwork = workersOutput.peek();
			population = repopulate(config, workersOutput);
			if ((gen % GENERATIONAL_DEBUG_INTERVAL) == 0) {
				System.out.print(" - Top 10: [");
				GenticNeuralNetwork<String>[] topNetworkArray = population
						.toArray(new ClassificationGenticNeuralNetwork[] {});
				for (int i = 0; i < 10; i++) {
					if (i > 0) {
						System.out.print(", ");
					}
					System.out.print(String.format("%.5f",topNetworkArray[i].getAverageError()));
				}
				System.out.println("]");
				if (config.getVisualizer() != null) {
					config.getVisualizer().drawNetwork(topNetworkArray[0], gen);
					config.getVisualizer().repaint();
				}
			}
		}
		((ClassificationGenticNeuralNetwork) bestNetwork).test(data, classLabels, cf, testingIndices);
		cf.printConfusionMatix();
		return (GenticNeuralNetwork<String>) bestNetwork;

	}

	/**
	 * Create a valid, randomized network.
	 */
	public void init(int numHiddenLayers, int numNeuronsPerLayer) {
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
			getLayers().add(currentLayerIds);
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
		getLayers().add(currentLayerIds);

		// create output layer
		createOutputLayer(previousLayer);
	}

	/**
	 * Create a valid output layer with one neuron for each class label.
	 * 
	 * @param previousLayer
	 *            the layer before the output layer
	 */
	protected void createOutputLayer(Set<Neuron> previousLayer) {
		Set<Neuron> currentLayer = new HashSet<Neuron>();
		Set<String> currentLayerIds = new HashSet<String>();
		uniqueClassLabels = calculateUniqueClassLabels(getY());
		outputs = new HashMap<OutputNeuron, String>();
		String[] lableStrings = uniqueClassLabels.toArray(new String[] {});
		for (int i = 0; i < uniqueClassLabels.size(); i++) {
			ContinuousOutputNeuron output = new ContinuousOutputNeuron(this);
			addOutput(lableStrings[i], output);
			currentLayer.add(output);
			currentLayerIds.add(output.getUuid());
		}
		for (Neuron prev : previousLayer) {
			for (Neuron next : currentLayer) {
				prev.addNext(next, 0.0);
			}
		}
		getLayers().add(currentLayerIds);
	}

	/**
	 * Create a unique set of output class labels.
	 * 
	 * @param classLabels
	 *            all class labels in the example data
	 * @return a set of output class labels
	 */
	protected static HashSet<String> calculateUniqueClassLabels(String[] classLabels) {
		HashSet<String> uniqueClassLabels = new HashSet<String>();
		for (String classLabel : classLabels) {
			uniqueClassLabels.add(classLabel);
		}
		return uniqueClassLabels;
	}

	public void test(double[][] inputData, String[] expected, ConfusionMatrix cf, List<Integer> testingIndices) {
		for (Integer testingIndex : testingIndices) {
			setupForPredict(inputData[testingIndex]);
			double highestProb = 0.0;
			String highestProbClass = null;
			for (Neuron output : outputs.keySet()) {
				double prob = output.activation();
				if (prob > highestProb) {
					highestProb = prob;
					highestProbClass = outputs.get(output);
				}
			}
			String expectedClass = expected[testingIndex];
			System.out.println("Expected " + expectedClass + ", predicted " + highestProbClass + " with probability "
					+ highestProb);
			cf.getConfusionMatrix()[cf.getClassLabelConfusionMatrixIndices().get(expectedClass)][cf
					.getClassLabelConfusionMatrixIndices().get(highestProbClass)]++;
		}
	}

	@Override
	public double error(double[] inputData, String expected) {
		double totalError = 0.0;
		setupForPredict(inputData);
		for (Neuron output : outputs.keySet()) {
			double prob = output.activation();
			double error = 0.0;
			if (outputs.get(output).equals(expected)) {
				error = Math.abs(1.0 - prob);
			} else {
				error = Math.abs(0.0 - prob);
			}
			totalError += error;
		}
		return totalError;
	}

	// FIXME figure out how to return both class and probability
	public String predict(double[] inputData) {
		double highestProb = 0.0;
		Neuron highestProbNeuron = null;
		setupForPredict(inputData);
		for (Neuron output : outputs.keySet()) {
			double prob = output.activation();
			if (prob > highestProb) {
				highestProbNeuron = output;
				highestProb = prob;
			}
		}
		System.out.println("Predicted class is " + outputs.get(highestProbNeuron) + " with prob value " + highestProb);
		return outputs.get(highestProbNeuron);
	}

	/**
	 * Clear the memoized values of all neurons and set the inputs.
	 * 
	 * @param inputData
	 *            the inputs to the network
	 */
	protected void setupForPredict(double[] inputData) {
		for (Map.Entry<String, Neuron> neuronEntry : getNeurons().entrySet()) {
			neuronEntry.getValue().resetMemoization();
		}
		for (int i = 0; i < inputData.length; i++) {
			InputNeuron in = featureToInputMap.get(i);
			in.setInput(inputData[i]);
		}
	}

	protected Neuron createNewRandomNeuron() {
		Neuron n = new ContinuousHiddenNeuron(this);
		neurons.put(n.getUuid(), n);
		return n;
	}

}
