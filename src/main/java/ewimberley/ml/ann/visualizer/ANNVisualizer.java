package ewimberley.ml.ann.visualizer;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.Ellipse2D;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.swing.JPanel;

import ewimberley.ml.ann.NeuralNetwork;
import ewimberley.ml.ann.Neuron;

public class ANNVisualizer extends JPanel {
	private static final String OUTPUT = "OUTPUT";
	private static final String INPUT = "INPUT";
	private static final String HIDDEN = "HIDDEN";
	private static final int NODE_DIAMETER = 60;
	private static final int MIN_X_SEPERATION = NODE_DIAMETER + 50;
	private static final int MIN_Y_SEPERATION = NODE_DIAMETER + 30;

	private Graphics2D g2d;
	private Map<String, NeuronCircle> neurons;
	int height, width;

	public ANNVisualizer(int width, int height) {
		neurons = new HashMap<String, NeuronCircle>();
		this.height = height;
		this.width = width;
	}

	private void doDrawing(Graphics g) {
		g2d = (Graphics2D) g;
		g2d.setBackground(new Color(255, 255, 255));
		g2d.clearRect(0, 0, width, height);
		g2d.setStroke(new BasicStroke(2));
		RenderingHints rh = new RenderingHints(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		rh.put(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
		g2d.setRenderingHints(rh);
		Color black = new Color(0, 0, 0);
		Color grey = new Color(150, 150, 150);
		Color green = new Color(0, 150, 0);
		Color blue = new Color(0, 0, 150);
		Color currentColor = black;
		// draw edges
		g2d.setPaint(grey);
		for (Map.Entry<String, NeuronCircle> neuronEntry : neurons.entrySet()) {
			NeuronCircle neuron = neuronEntry.getValue();
			for (String linkId : neuron.getLinks()) {
				NeuronCircle other = neurons.get(linkId);
				int radius = NODE_DIAMETER / 2;
				g2d.drawLine(neuron.getX() + radius, neuron.getY() + radius, other.getX() + radius,
						other.getY() + radius);
			}
		}
		// draw nodes
		for (Map.Entry<String, NeuronCircle> neuronEntry : neurons.entrySet()) {
			NeuronCircle neuron = neuronEntry.getValue();
			if (neuron.getType().equals(HIDDEN)) {
				currentColor = black;
			} else if (neuron.getType().equals(INPUT)) {
				currentColor = green;
			} else {
				currentColor = blue;
			}
			g2d.setPaint(currentColor);
			g2d.fillOval(neuron.getX(), neuron.getY(), NODE_DIAMETER, NODE_DIAMETER);
		}
	}

	public void drawNetwork(NeuralNetwork network) {
		Map<String, NeuronCircle> newNeurons = new HashMap<String, NeuronCircle>();
		List<Set<String>> layers = network.getLayers();
		int usableHeight = height - 2 * (MIN_X_SEPERATION);
		int middleY = usableHeight / 2;
		int currentX = MIN_X_SEPERATION;
		int currentY = MIN_Y_SEPERATION;
		for (int i = 0; i < layers.size(); i++) {
			Set<String> layer = layers.get(i);
			int numNodesInLayer = layer.size();
			int seperation = usableHeight / numNodesInLayer;
			String neuronType = HIDDEN;
			if (i == 0) {
				neuronType = INPUT;
			} else if ((i + 1) == layers.size()) {
				neuronType = OUTPUT;
			}
			currentX += MIN_X_SEPERATION + NODE_DIAMETER;
			currentY = middleY - (numNodesInLayer * MIN_Y_SEPERATION / 2);
			// for (String neuron : layer) {
			String[] neurons = layer.toArray(new String[] {});
			for (int j = 0; j < neurons.length; j++) {
				String neuron = neurons[j];
				NeuronCircle neuronShape = new NeuronCircle(neuron, neuronType, currentX, currentY);
				newNeurons.put(neuron, neuronShape);
				Neuron neuronObj = (Neuron) network.getNeurons().get(neuron);
				for (String linkId : neuronObj.getNext()) {
					neuronShape.addLink(linkId);
				}
				currentY += MIN_Y_SEPERATION;
			}
		}
		neurons = newNeurons;
	}

	@Override
	public void paintComponent(Graphics g) {

		super.paintComponent(g);
		doDrawing(g);
	}

}
