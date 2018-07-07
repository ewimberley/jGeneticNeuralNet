package ewimberley.ml.examples;

import javax.swing.JFrame;

import ewimberley.ml.DataLoader;
import ewimberley.ml.ann.NeuralNetworkTrainingConfiguration;
import ewimberley.ml.ann.gnn.GenticNeuralNetwork;
import ewimberley.ml.ann.gnn.classifier.ClassificationGenticNeuralNetwork;
import ewimberley.ml.ann.visualizer.ANNVisualizer;

public class YeastGNNExample extends JFrame {

	private static final int HEIGHT = 1600;
	private static final int WIDTH = 1600;
	private ANNVisualizer vis;

	public YeastGNNExample() {
		initUI();
	}

	public static void main(String[] args) {
		YeastGNNExample main = new YeastGNNExample();
		main.setVisible(true);
		main.train();
	}

	private void train() {
		//FIXME add citation to dataset
		String dataFile = "src/test/resources/yeast.data";
		DataLoader dl = new DataLoader();
		dl.loadCSVFile(dataFile);
		NeuralNetworkTrainingConfiguration config = new NeuralNetworkTrainingConfiguration();
		config.setNumNetworksPerGeneration(300);
		config.setNumGenerations(5000);
		config.setNumHiddenLayers(4);
		config.setNumNeuronsPerLayer(14);
		config.setMaxLearningRate(4.0);
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
