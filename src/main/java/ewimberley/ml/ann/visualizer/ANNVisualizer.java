package ewimberley.ml.ann.visualizer;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.geom.Ellipse2D;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JTextField;

import ewimberley.ml.ann.InputNeuron;
import ewimberley.ml.ann.NeuralNetwork;
import ewimberley.ml.ann.Neuron;
import ewimberley.ml.ann.gnn.classifier.ClassificationGenticNeuralNetwork;

public class ANNVisualizer extends JPanel {
	private static final String OUTPUT = "OUTPUT";
	private static final String INPUT = "INPUT";
	private static final String HIDDEN = "HIDDEN";
	private static final int NODE_DIAMETER = 60;
	private static final int MIN_X_SEPERATION = NODE_DIAMETER + 50;
	private static final int MIN_Y_SEPERATION = NODE_DIAMETER + 50;

	private static final Color BLACK = new Color(0, 0, 0);
	private static final Color GREY = new Color(125, 125, 125);
	private static final Color GREEN = new Color(0, 150, 0);
	private static final Color BLUE = new Color(0, 0, 150);
	private static final Color WHITE = new Color(255, 255, 255);

	private Graphics2D g2d;
	private Map<String, NeuronCircle> neurons;
	private Map<String, NeuronCircle> latest;
	private Map<String, NeuronCircle> latestDrawn;
	private NeuralNetwork<?> network;
	int displayNetwork;
	boolean manualMode;
	private List<Map<String, NeuronCircle>> previousNetworks;
	private List<Integer> generationNums;
	int height, width;
	private PredictAction predict;
	private JButton predictButton;
	private List<JTextField> inputs;
	private boolean inputsInit;

	public ANNVisualizer(int width, int height) {
		neurons = new HashMap<String, NeuronCircle>();
		previousNetworks = new ArrayList<Map<String, NeuronCircle>>();
		generationNums = new ArrayList<Integer>();
		displayNetwork = -1;
		this.height = height;
		this.width = width;
		predictButton = new JButton("Predict");
		predictButton.setBounds(10, (width / 2 - 50), 100, 25);
		predictButton.removeActionListener(predict);
		predict = new PredictAction(this);
		predictButton.addActionListener(predict);
		this.add(predictButton);
		inputs = new ArrayList<JTextField>();
		inputsInit = false;
	}

	private void doDrawing(Graphics g) {
		g2d = (Graphics2D) g;

		int radius = NODE_DIAMETER / 2;
		g2d.setBackground(WHITE);
		g2d.clearRect(0, 0, width, height);
		// g2d.setStroke(new BasicStroke(2));
		RenderingHints rh = new RenderingHints(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		rh.put(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
		g2d.setRenderingHints(rh);

		Color currentColor = BLACK;
		if (displayNetwork > 0) {
			g2d.drawString("Generation number: " + generationNums.get(displayNetwork - 1), 10, 10);
		}
		// draw edges
		Map<String, Integer> weightYOffsets = new HashMap<String, Integer>();
		for (Map.Entry<String, NeuronCircle> neuronEntry : neurons.entrySet()) {
			NeuronCircle neuron = neuronEntry.getValue();
			for (String linkId : neuron.getLinks()) {
				if (!weightYOffsets.containsKey(linkId)) {
					weightYOffsets.put(linkId, 0);
				} else {
					weightYOffsets.put(linkId, weightYOffsets.get(linkId) + 10);
				}
				NeuronCircle other = neurons.get(linkId);
				g2d.setPaint(GREY);
				g2d.drawLine(neuron.getX() + radius, neuron.getY() + radius, other.getX() + radius,
						other.getY() + radius);
				g2d.setPaint(BLACK);
				g2d.drawString(String.format("%.2f", neuron.getWeights().get(linkId)), other.getX() - NODE_DIAMETER,
						other.getY() + weightYOffsets.get(linkId));
			}
		}
		// draw nodes
		for (Map.Entry<String, NeuronCircle> neuronEntry : neurons.entrySet()) {
			NeuronCircle neuron = neuronEntry.getValue();
			if (neuron.getType().equals(HIDDEN)) {
				currentColor = BLACK;
			} else if (neuron.getType().equals(INPUT)) {
				currentColor = GREEN;
				if (!inputsInit) {
					JTextField textField = new JTextField();
					textField.setBounds(neuron.getX() - 110, neuron.getY() + 12, 100, 25);
					this.add(textField);
					inputs.add(textField);
				}
			} else {
				currentColor = BLUE;
			}
			g2d.setPaint(currentColor);
			g2d.fillOval(neuron.getX(), neuron.getY(), NODE_DIAMETER, NODE_DIAMETER);
			g2d.setPaint(WHITE);
			g2d.drawString(String.format("%.2f", neuron.getBias()), neuron.getX() + 5, neuron.getY() + radius);
			if (neuron.getAct() != null) {
				g2d.drawString(neuron.getAct().name(), neuron.getX() + 5, neuron.getY() + radius + 10);
			}
			if (neuron.getClassLabel() != null) {
				g2d.setPaint(BLACK);
				g2d.drawString(neuron.getClassLabel(), neuron.getX() + NODE_DIAMETER + 5, neuron.getY() + radius + 10);
			}
		}
		if (!inputsInit && !inputs.isEmpty()) {
			inputsInit = true;
		}

	}

	public void drawNetwork(NeuralNetwork network, int generationNum) {
		this.setNetwork(network);
		Map<String, NeuronCircle> newNeurons = new HashMap<String, NeuronCircle>();
		List<Set<String>> layers = network.getLayers();
		int usableHeight = height - 2 * (MIN_X_SEPERATION);
		int middleY = usableHeight / 2;
		int currentX = MIN_X_SEPERATION;
		int currentY = MIN_Y_SEPERATION;
		for (int i = 0; i < layers.size(); i++) {
			Set<String> layer = layers.get(i);
			int numNodesInLayer = layer.size();
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
				if (network instanceof ClassificationGenticNeuralNetwork) {
					if ((i + 1) == layers.size()) {
						String classLabel = (String) network.getOutputs().get(neuronObj);
						neuronShape.setClassLabel(classLabel);
					}
				}
				neuronShape.setBias(neuronObj.getBias());
				neuronShape.setAct(neuronObj.getActivationFunction());
				for (String linkId : neuronObj.getNext()) {
					neuronShape.addLink(linkId);
					neuronShape.addWeight(linkId, neuronObj.getNextWeights().get(linkId));
				}
				currentY += MIN_Y_SEPERATION;
			}
		}

		if (latest != null) {
			previousNetworks.add(latest);
		}
		neurons = newNeurons;
		generationNums.add(generationNum);
		latest = newNeurons;
	}

	@Override
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		doDrawing(g);
	}

	public NeuralNetwork<?> getNetwork() {
		return network;
	}

	public void setNetwork(NeuralNetwork<?> network) {
		this.network = network;
	}

	public List<JTextField> getInputs() {
		return inputs;
	}

}
