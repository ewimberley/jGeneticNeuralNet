package ewimberley.ml.gnn.classifier;

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
import ewimberley.ml.gnn.ContinuousHiddenNeuron;
import ewimberley.ml.gnn.ContinuousOutputNeuron;
import ewimberley.ml.gnn.GenticNeuralNetwork;
import ewimberley.ml.gnn.InputNeuron;
import ewimberley.ml.gnn.NeuralNetwork;
import ewimberley.ml.gnn.NeuralNetworkErrorComparator;
import ewimberley.ml.gnn.Neuron;
import ewimberley.ml.gnn.OutputNeuron;

public class ClassificationGenticNeuralNetwork extends GenticNeuralNetwork<Double> {
	
	public ClassificationGenticNeuralNetwork(double[][] data, String[] classLabels) {
		super(data, classLabels);
	}

	public ClassificationGenticNeuralNetwork(NeuralNetwork<Double> toClone) {
		super(toClone.getData(), toClone.getClassLabels());
		for (Map.Entry<String, Neuron<Double>> neuronEntry : toClone.getNeurons().entrySet()) {
			String id = neuronEntry.getKey();
			Neuron<Double> neuron = neuronEntry.getValue();
			if (neuron instanceof InputNeuron) {
				InputNeuron<Double> clone = new InputNeuron<Double>(this, (InputNeuron<Double>) neuron);
				addInput(clone);
			} else if (neuron instanceof OutputNeuron) {
				OutputNeuron<Double> clone = new ContinuousOutputNeuron(this, (ContinuousOutputNeuron) neuron);
				String classLabel = toClone.getOutputs().get(neuron);
				addOutput(classLabel, clone);
			} else {
				Neuron<Double> clone = new ContinuousHiddenNeuron(this, (ContinuousHiddenNeuron) neuron);
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

	public static Learner train(double[][] data, String[] classLabels, int numNetworksPerGeneration,
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
		PriorityQueue<ClassificationGenticNeuralNetwork> population = new PriorityQueue<ClassificationGenticNeuralNetwork>(
				new NeuralNetworkErrorComparator<Double>());
		for (int i = 0; i < numNetworksPerGeneration; i++) {
			ClassificationGenticNeuralNetwork network = new ClassificationGenticNeuralNetwork(data, classLabels);
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
			PriorityQueue<ClassificationGenticNeuralNetwork> survivors = new PriorityQueue<ClassificationGenticNeuralNetwork>(
					new NeuralNetworkErrorComparator<Double>());
			ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
			List<ClassificationGenticNeuralNetworkWorker> workers = new ArrayList<ClassificationGenticNeuralNetworkWorker>();
			for (int onNetwork = 0; onNetwork < numNetworksPerGeneration; onNetwork++) {
				ClassificationGenticNeuralNetwork network = population.poll();
				// System.out.println("Evaluating network w/ avg error: " +
				// network.getAverageError());
				ClassificationGenticNeuralNetworkWorker worker = new ClassificationGenticNeuralNetworkWorker(network, data, classLabels, trainingIndices);
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
			for (ClassificationGenticNeuralNetworkWorker worker : workers) {
				ClassificationGenticNeuralNetwork mutant = (ClassificationGenticNeuralNetwork) worker.getMutant();
				double averageMutantError = mutant.getAverageError();
				ClassificationGenticNeuralNetwork original = (ClassificationGenticNeuralNetwork) worker.getOriginal();
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
		((ClassificationGenticNeuralNetwork)bestNetwork).test(data, classLabels, cf, testingIndices);
		cf.printConfusionMatix();
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
	
	protected void createOutputLayer(Set<Neuron<Double>> previousLayer, Set<Neuron<Double>> currentLayer, Set<String> currentLayerIds) {
		uniqueClassLabels = calculateUniqueClassLabels(getClassLabels());
		outputs = new HashMap<OutputNeuron<Double>, String>();
		String[] lableStrings = uniqueClassLabels.toArray(new String[] {});
		for (int i = 0; i < uniqueClassLabels.size(); i++) {
			ContinuousOutputNeuron output = new ContinuousOutputNeuron(this);
			addOutput(lableStrings[i], output);
			currentLayer.add(output);
			currentLayerIds.add(output.getUuid());
		}
		for (Neuron<Double> prev : previousLayer) {
			for (Neuron<Double> next : currentLayer) {
				prev.addNext(next, 0.0);
			}
		}
	}
	
	public void test(double[][] inputData, String[] expected, ConfusionMatrix cf, List<Integer> testingIndices) {
		for (Integer testingIndex : testingIndices) {
			setupForPredict(inputData[testingIndex]);
			double highestProb = 0.0;
			String highestProbClass = null;
			for (Neuron<Double> output : outputs.keySet()) {
				double prob = output.activation();
				if(prob > highestProb) {
					highestProb = prob;
					highestProbClass = outputs.get(output);
				}
			}
			String expectedClass = expected[testingIndex];
			System.out.println("Expected " + expectedClass + ", predicted " + highestProbClass + " with probability " + highestProb);
			cf.getConfusionMatrix()[cf.getClassLabelConfusionMatrixIndices().get(expectedClass)][cf.getClassLabelConfusionMatrixIndices().get(highestProbClass)]++;
		}
	}

	public double error(double[] inputData, String expected) {
		double totalError = 0.0;
		setupForPredict(inputData);
		for (Neuron<Double> output : outputs.keySet()) {
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
	
	// FIXME figure out how to return class and probability
	public String predict(double[] inputData) {
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
		return outputs.get(highestProbNeuron);
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
