package ewimberley.ml.examples;

import javax.swing.JFrame;

import ewimberley.ml.DataLoader;
import ewimberley.ml.ann.NeuralNetworkTrainingConfiguration;
import ewimberley.ml.ann.gnn.GenticNeuralNetwork;
import ewimberley.ml.ann.gnn.classifier.ClassificationGenticNeuralNetwork;
import ewimberley.ml.ann.visualizer.ANNVisualizer;

public class IrisGNNExample extends JFrame {
	
	private static final int HEIGHT = 1600;
	private static final int WIDTH = 1600;
	private ANNVisualizer vis;

	public IrisGNNExample() {
		initUI();
	}

	public static void main(String[] args) {
		IrisGNNExample main = new IrisGNNExample();
		main.setVisible(true);
		main.train();
	}

	private void train() {
		/*
		 * Lichman, M. (2013). UCI Machine Learning Repository
		 * [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School
		 * of Information and Computer Science.
		 */
		String dataFile = "src/test/resources/iris.data";
		DataLoader dl = new DataLoader();
		dl.loadCSVFile(dataFile);
		NeuralNetworkTrainingConfiguration config = new NeuralNetworkTrainingConfiguration();
		config.setNumNetworksPerGeneration(1000);
		config.setNumGenerations(100);
		config.setNumHiddenLayers(5);
		config.setNumNeuronsPerLayer(8);
		config.setMaxLearningRate(100.0);
		config.setVisualizer(vis);
		GenticNeuralNetwork<String> model = ClassificationGenticNeuralNetwork.train(dl.getData(),
				//dl.getClassLabels(), 500, 300, 5, 8, 100.0);
				dl.getClassLabels(), config);
		model.printNetwork();
	}
	
	private void initUI() {
		vis = new ANNVisualizer(WIDTH, HEIGHT);
		add(vis);

		setTitle("Yeast Neural Network Visualizer");
		setSize(WIDTH, HEIGHT);
		setLocationRelativeTo(null);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

}
