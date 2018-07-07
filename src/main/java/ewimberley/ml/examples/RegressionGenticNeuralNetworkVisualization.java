package ewimberley.ml.examples;

import java.awt.EventQueue;

import javax.swing.JFrame;

import org.junit.Test;

import ewimberley.ml.ann.NeuralNetworkTrainingConfiguration;
import ewimberley.ml.ann.gnn.regression.RegressionGenticNeuralNetwork;
import ewimberley.ml.ann.visualizer.ANNVisualizer;

public class RegressionGenticNeuralNetworkVisualization extends JFrame {
	
	private static final int HEIGHT = 1200;
	private static final int WIDTH = 1600;
	private ANNVisualizer vis;

	public RegressionGenticNeuralNetworkVisualization() {
		initUI();
	}

	public static void main(String[] args) {
		RegressionGenticNeuralNetworkVisualization main = new RegressionGenticNeuralNetworkVisualization();
		main.setVisible(true);
		main.train();
	}
	
	private void initUI() {
		vis = new ANNVisualizer(WIDTH, HEIGHT);
		add(vis);

		setTitle("Regression Neural Network Visualization");
		setSize(WIDTH, HEIGHT);
		setLocationRelativeTo(null);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}
	
	private void train() {
		int numX = 100;
		double[][] data = new double[numX][1];
		double[] values = new double[numX];
		for (int i = 0; i < numX; i++) {
			data[i][0] = i;
			values[i] = i * i;
		}				
		NeuralNetworkTrainingConfiguration config = new NeuralNetworkTrainingConfiguration();
		config.setNumNetworksPerGeneration(1000);
		config.setNumGenerations(1000);
		config.setNumHiddenLayers(2);
		config.setNumNeuronsPerLayer(6);
		config.setMaxLearningRate(1.0);
		config.setVisualizer(vis);
		config.setProbMutateActivationFunction(0.1);
		RegressionGenticNeuralNetwork bestNetwork = (RegressionGenticNeuralNetwork) RegressionGenticNeuralNetwork.train(data, values, config);
		// assertTrue(bestNetwork.getAverageError() < 1.0);
		System.out.println(bestNetwork.getAverageError());

		// for(int i = 0; i < numX; i++) {
		for (int i = 0; i < 5; i++) {
			// tolerance of 1%
			System.out.println(i + "\t" + values[i] + "\t" + bestNetwork.predict(new double[] { (double) i }));
			// assertEquals(values[i], bestNetwork.predict(new double[] {(double)i}),
			// (values[i]/100.0));
		}
	}

}
