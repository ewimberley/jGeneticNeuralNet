package ewimberley.ml.examples;

import javax.swing.JFrame;

import ewimberley.ml.DataLoader;
import ewimberley.ml.ann.gnn.GenticNeuralNetwork;
import ewimberley.ml.ann.gnn.GeneticNeuralNetworkTrainingConfiguration;
import ewimberley.ml.ann.gnn.classifier.ClassificationGenticNeuralNetwork;
import ewimberley.ml.ann.visualizer.ANNVisualizer;

public class TitanicGNNExample extends JFrame {
	
	private static final int HEIGHT = 1600;
	private static final int WIDTH = 1600;
	private ANNVisualizer vis;

	public TitanicGNNExample() {
		initUI();
	}

	public static void main(String[] args) {
		TitanicGNNExample main = new TitanicGNNExample();
		main.setVisible(true);
		main.train();
	}

	private void train() {
		/*
		 * https://www.kaggle.com/shisancd/titanic
		 */
		String dataFile = "src/test/resources/titanic.csv";
		DataLoader dl = new DataLoader();
		dl.loadCSVFile(dataFile);
		GeneticNeuralNetworkTrainingConfiguration config = new GeneticNeuralNetworkTrainingConfiguration();
		config.setNumNetworksPerGeneration(500);
		config.setNumGenerations(2000);
		config.setNumHiddenLayers(4);
		config.setNumNeuronsPerLayer(6);
		config.setMaxLearningRate(20.0);
		config.setMaxThreads(Runtime.getRuntime().availableProcessors() * 2);
		config.setVisualizer(vis);
		GenticNeuralNetwork<String> model = ClassificationGenticNeuralNetwork.train(dl.getData(), dl.getClassLabels(), config);
		model.printNetwork();
	}
	
	private void initUI() {
		vis = new ANNVisualizer(WIDTH, HEIGHT);
		add(vis);
		setTitle("Iris Neural Network Visualizer");
		setSize(WIDTH, HEIGHT);
		setLocationRelativeTo(null);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

}
