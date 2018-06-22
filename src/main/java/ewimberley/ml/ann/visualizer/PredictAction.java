package ewimberley.ml.ann.visualizer;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JTextField;

import ewimberley.ml.ann.NeuralNetwork;
import ewimberley.ml.ann.gnn.regression.RegressionGenticNeuralNetwork;

public class PredictAction implements ActionListener {
	
	private ANNVisualizer vis;
	
	public PredictAction(ANNVisualizer vis) {
		this.vis = vis;
	}

	public void actionPerformed(ActionEvent e) {
		List<JTextField> inputs = vis.getInputs();
		double[] inputValues = new double[inputs.size()];
		int onInput = 0;
		for(JTextField input : inputs) {
			inputValues[onInput] = Double.parseDouble(input.getText());
			onInput++;
		}
		NeuralNetwork<?> network = vis.getNetwork();
		if(network instanceof RegressionGenticNeuralNetwork) {
			Object output = ((RegressionGenticNeuralNetwork)network).predict(inputValues);
			System.out.println(output);
		}
	}

}
