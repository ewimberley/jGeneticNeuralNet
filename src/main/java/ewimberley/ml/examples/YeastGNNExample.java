package ewimberley.ml.examples;

import javax.swing.JFrame;

import ewimberley.ml.DataLoader;
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
		/*
		 * Lichman, M. (2013). UCI Machine Learning Repository
		 * [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School
		 * of Information and Computer Science.
		 */
		String dataFile = "src/test/resources/yeast.data";
		DataLoader dl = new DataLoader();
		dl.loadCSVFile(dataFile);
		ClassificationGenticNeuralNetwork model = ClassificationGenticNeuralNetwork.train(dl.getData(),
				dl.getClassLabels(), 100, 1000, 5, 12, 100.0, vis);
		model.printNetwork();
	}
	
	private void initUI() {
		vis = new ANNVisualizer(WIDTH, HEIGHT);
		add(vis);

		setTitle("Basic shapes");
		setSize(WIDTH, HEIGHT);
		setLocationRelativeTo(null);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

}
