package ewimberley.gnn;

public class NeuralNetworkWorker implements Runnable {
	
	private NeuralNetwork network;
	
	private double[] data;
	
	private String expected;
	
	private double error;
	
	public NeuralNetworkWorker(NeuralNetwork network, double[] data, String expected) {
		this.network = network;
		this.data = data;
		this.expected = expected;
	}
	
    public void run() {
    	error = network.error(data, expected);
    	//System.out.println("Found error: " + error);
    }

	public double getError() {
		return error;
	}

}
