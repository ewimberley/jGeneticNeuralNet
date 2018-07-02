package ewimberley.ml.ann.gnn.regression;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ewimberley.ml.Learner;
import ewimberley.ml.ann.ActivationFunction;
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
import ewimberley.ml.ann.gnn.classifier.ClassificationGenticNeuralNetwork;
import ewimberley.ml.ann.gnn.classifier.ClassificationGenticNeuralNetworkWorker;
import ewimberley.ml.ann.visualizer.ANNVisualizer;

public class RegressionGenticNeuralNetwork extends GenticNeuralNetwork<Double> {

	private static final int GENERATIONAL_DEBUG_INTERVAL = 10;
	private OutputNeuron output;

	public RegressionGenticNeuralNetwork(double[][] data, Double[] y) {
		super(data, y);
	}

	public RegressionGenticNeuralNetwork(RegressionGenticNeuralNetwork toClone) {
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
				output = clone;
				outputs.put(output, 0.0);
				neurons.put(output.getUuid(), output);
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

	public static Learner<Double> train(double[][] data, double[] yPrim, NeuralNetworkTrainingConfiguration config) {
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

		GenticNeuralNetwork<?> bestNetwork = null;
		PriorityQueue<GenticNeuralNetwork<?>> population = new PriorityQueue<GenticNeuralNetwork<?>>(
				new GenticNeuralNetworkErrorComparator());
		for (int i = 0; i < config.getNumNetworksPerGeneration(); i++) {
			RegressionGenticNeuralNetwork network = new RegressionGenticNeuralNetwork(data, y);
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
			if (numChildrenPerNetwork < 50.0) {
				numChildrenPerNetwork = 50.0;
			}
			while (numNetworks < config.getNumNetworksPerGeneration()) {
				if (numChildrenPerNetwork > 1) {
					numChildrenPerNetwork -= 5.0;
				} else {
					numChildrenPerNetwork = 1.0;
				}
				int intNumChildrenPerNetwork = (int) Math.ceil(numChildrenPerNetwork);
				RegressionGenticNeuralNetwork network = (RegressionGenticNeuralNetwork) population.poll();
				for (int onChild = 0; onChild < intNumChildrenPerNetwork; onChild++) {
					RegressionGenticNeuralNetworkWorker worker = new RegressionGenticNeuralNetworkWorker(network, data,
							y, trainingIndices);
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
				RegressionGenticNeuralNetwork[] topNetworkArray = population
						.toArray(new RegressionGenticNeuralNetwork[] {});
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
		double mse = ((RegressionGenticNeuralNetwork) bestNetwork).test(data, y, testingIndices);
		System.out.println("Mean squared error: " + mse);
		return (Learner<Double>) bestNetwork;
	}

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
		currentLayer = new HashSet<Neuron>();
		currentLayerIds = new HashSet<String>();

		// create output layer
		createOutputLayer(previousLayer, currentLayer, currentLayerIds);
	}

	protected void createOutputLayer(Set<Neuron> previousLayer, Set<Neuron> currentLayer, Set<String> currentLayerIds) {
		output = new ContinuousOutputNeuron(this);
		((ContinuousOutputNeuron) output).setActivationFunction(ActivationFunction.LINEAR);
		neurons.put(output.getUuid(), output);
		currentLayer.add(output);
		currentLayerIds.add(output.getUuid());
		outputs.put(output, 0.0);
		for (Neuron prev : previousLayer) {
			for (Neuron next : currentLayer) {
				prev.addNext(next, 0.0);
			}
		}
		getLayers().add(currentLayerIds);
	}

	public double test(double[][] inputData, Double[] expected, List<Integer> testingIndices) {
		double mse = 0.0;
		for (Integer testingIndex : testingIndices) {
			setupForPredict(inputData[testingIndex]);
			double predict = output.activation();
			double error = Math.abs(expected[testingIndex] - predict);
			mse += error * error;
		}
		mse /= testingIndices.size();
		return mse;
	}

	@Override
	public double error(double[] inputData, Double expected) {
		setupForPredict(inputData);
		double predict = output.activation();
		double error = Math.abs(expected - predict);
		return error;
	}

	public Double predict(double[] inputData) {
		setupForPredict(inputData);
		// System.out.println("Predicted class is " + outputs.get(highestProbNeuron) + "
		// with prob value " + highestProb);
		return output.activation();
	}

	protected void setupForPredict(double[] inputData) {
		for (Map.Entry<String, Neuron> neuronEntry : getNeurons().entrySet()) {
			neuronEntry.getValue().resetMemoization();
		}
		output.resetMemoization();
		for (int i = 0; i < inputData.length; i++) {
			InputNeuron in = featureToInputMap.get(i);
			in.setInput(inputData[i]);
		}
	}

	protected Neuron createNewRandomNeuron() {
		ContinuousHiddenNeuron n = new ContinuousHiddenNeuron(this);
		n.setActivationFunction(ActivationFunction.LINEAR);
		neurons.put(n.getUuid(), n);
		return n;
	}

	public OutputNeuron getOutput() {
		return output;
	}

	public void setOutput(OutputNeuron output) {
		this.output = output;
	}

}
