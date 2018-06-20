package ewimberley.ml.ann.visualizer;

import java.util.ArrayList;
import java.util.List;

public class NeuronCircle {
	
	private String id;
	
	private List<String> links;
	
	private int x, y;
	
	private String type;
	
	public NeuronCircle(String id, String type, int x, int y) {
		this.links = new ArrayList<String>();
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
	
}
