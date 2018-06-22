package ewimberley.ml.ann.visualizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ewimberley.ml.ann.ActivationFunction;

public class NeuronCircle {
	
	private String id;
	
	private double bias;
	
	private ActivationFunction act;
	
	private List<String> links;
	
	private Map<String, Double> weights;
	
	private int x, y;
	
	private String type;
	
	private String classLabel;
	
	public NeuronCircle(String id, String type, int x, int y) {
		this.links = new ArrayList<String>();
		this.weights = new HashMap<String, Double>();
		this.type = type;
		this.id = id;
		this.x = x;
		this.y = y;
	}

	public int getX() {
		return x;
	}

	public void setX(int x) {
		this.x = x;
	}

	public int getY() {
		return y;
	}

	public void setY(int y) {
		this.y = y;
	}
	
	public void addLink(String id) {
		this.getLinks().add(id);
	}

	public List<String> getLinks() {
		return links;
	}

	public String getType() {
		return type;
	}

	public void setType(String type) {
		this.type = type;
	}

	public String getId() {
		return id;
	}

	public double getBias() {
		return bias;
	}

	public void setBias(double bias) {
		this.bias = bias;
	}

	public ActivationFunction getAct() {
		return act;
	}

	public void setAct(ActivationFunction act) {
		this.act = act;
	}

	public Map<String, Double> getWeights() {
		return weights;
	}

	public void addWeight(String id, double weight) {
		this.weights.put(id, weight);
	}

	public String getClassLabel() {
		return classLabel;
	}

	public void setClassLabel(String classLabel) {
		this.classLabel = classLabel;
	}
	
}
