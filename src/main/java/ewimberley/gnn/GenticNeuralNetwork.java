package ewimberley.gnn;

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

public class GenticNeuralNetwork extends NeuralNetwork {

	private static final int NUM_THREADS = 100;
	
	private double averageError;
	
	public GenticNeuralNetwork(NeuralNetwork toClone) {
		this(toClone.data, toClone.classLabels);
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
			featureToInputMap.put(inputMappingEntry.getKey(), (InputNeuron) neurons.get(inputMappingEntry.getValue().getUuid()));
		}
		this.layers = toClone.layers;
		setLearningRate(toClone.getLearningRate() * (1.0 - toClone.getAnnealingRate()));
	}

	public GenticNeuralNetwork(double[][] data, String[] classLabels) {
		super();
		setLearningRate(0.4); // reasonable default
		setAnnealingRate(0.03);
		this.data = data;
		this.classLabels = classLabels;
		this.neurons = new HashMap<String, Neuron>();
		numHiddenLayers = 1; // reasonable default
		numNeuronsPerLayer = 5; // reasonable default
	}

	public static NeuralNetwork train(double[][] data, String[] classLabels, int numNetworksPerGeneration,
			int numGenerations, int numHiddenLayers, int numNeuronsPerLayer, double learningRate) {
		// FIXME 10-fold cross validation
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
		GenticNeuralNetwork bestNetwork = null;
		PriorityQueue<GenticNeuralNetwork> population = new PriorityQueue<GenticNeuralNetwork>(new NeuralNetworkErrorComparator());
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
			PriorityQueue<GenticNeuralNetwork> survivors = new PriorityQueue<GenticNeuralNetwork>(new NeuralNetworkErrorComparator());
			//for (GenticNeuralNetwork network : population) {
			for(int onNetwork = 0; onNetwork < numNetworksPerGeneration; onNetwork++) {
				GenticNeuralNetwork network = population.poll();
				System.out.println("Evaluating fitness of network with average error: " + network.getAverageError());
				ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
				List<NeuralNetworkWorker> originalWorkers = new ArrayList<NeuralNetworkWorker>();
				List<NeuralNetworkWorker> mutantWorkers = new ArrayList<NeuralNetworkWorker>();
				GenticNeuralNetwork clone = new GenticNeuralNetwork(network);
		    	clone.mutate();
				for (Integer trainingIndex : trainingIndices) {
					NeuralNetworkWorker originalWorker = new NeuralNetworkWorker(network, data[trainingIndex], classLabels[trainingIndex]);
					originalWorkers.add(originalWorker);
					executor.execute(originalWorker);
			    	NeuralNetworkWorker mutantworker = new NeuralNetworkWorker(clone, data[trainingIndex], classLabels[trainingIndex]);
			    	mutantWorkers.add(mutantworker);
					executor.execute(mutantworker);
				}
				executor.shutdown();
				while(!executor.isTerminated()) {
					try {
						Thread.sleep(25);
					} catch (InterruptedException e) {
						//do nothing
					}
				}
				//calculate original average error
				double averageOriginalError = 0.0;
				for(NeuralNetworkWorker worker : originalWorkers) {
					averageOriginalError += worker.getError();
				}
				averageOriginalError /= originalWorkers.size();
				network.setAverageError(averageOriginalError);
				survivors.add(network);

				//calculate mutant average error
				double averageMutantError = 0.0;
				for(NeuralNetworkWorker worker : mutantWorkers) {
					averageMutantError += worker.getError();
				}
				averageMutantError /= mutantWorkers.size();
				if(averageMutantError < averageOriginalError) {
					//System.out.println("Mutant is better by " + (averageOriginalError - averageMutantError));
					survivors.add(clone);
					clone.setAverageError(averageMutantError);
					if (averageMutantError < bestAverageError) {
						bestNetwork = (GenticNeuralNetwork) clone;
						bestAverageError = averageMutantError;
					}
				} else if (averageOriginalError < bestAverageError) {
					bestNetwork = network;
					bestAverageError = averageOriginalError;
				}
				
				
			}
			population = survivors;
			bestNetwork.printNetwork();
			System.out.println("Best network had average error: " + bestAverageError);
		}
		System.out.println("Best network had average error: " + bestAverageError);
		bestNetwork.printNetwork();
		for(Integer testingIndex : testingIndices) {
			bestNetwork.printError(data[testingIndex], classLabels[testingIndex]);
		}
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

		uniqueClassLabels = new HashSet<String>();
		for (String classLabel : classLabels) {
			uniqueClassLabels.add(classLabel);
		}
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
