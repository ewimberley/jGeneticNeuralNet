package ewimberley.ml.examples;

import javax.swing.JFrame;

import ewimberley.ml.DataLoader;
import ewimberley.ml.ann.NeuralNetworkTrainingConfiguration;
import ewimberley.ml.ann.gnn.GenticNeuralNetwork;
import ewimberley.ml.ann.gnn.classifier.ClassificationGenticNeuralNetwork;
import ewimberley.ml.ann.visualizer.ANNVisualizer;

public class EcoliGNNExample extends JFrame {

	private static final int HEIGHT = 1600;
	private static final int WIDTH = 1600;
	private ANNVisualizer vis;

	public EcoliGNNExample() {
		initUI();
	}

	public static void main(String[] args) {
		EcoliGNNExample main = new EcoliGNNExample();
		main.setVisible(true);
		main.train();
	}

	private void train() {
		//FIXME add citation to dataset
		String dataFile = "src/test/resources/ecoli.csv";
		DataLoader dl = new DataLoader();
		dl.loadCSVFile(dataFile);
		NeuralNetworkTrainingConfiguration config = new NeuralNetworkTrainingConfiguration();
		config.setNumNetworksPerGeneration(1000);
		config.setNumGenerations(500);
		config.setNumHiddenLayers(3);
		config.setNumNeuronsPerLayer(10);
		config.setMaxLearningRate(2.0);
		config.setVisualizer(vis);
		GenticNeuralNetwork<String> model = ClassificationGenticNeuralNetwork.train(dl.getData(),
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
