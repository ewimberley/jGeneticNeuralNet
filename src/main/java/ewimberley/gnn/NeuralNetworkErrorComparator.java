package ewimberley.gnn;

import java.util.Comparator;

public class NeuralNetworkErrorComparator implements Comparator<GenticNeuralNetwork> {

	public int compare(GenticNeuralNetwork c1, GenticNeuralNetwork c2) {
		if(c1.getAverageError() < c2.getAverageError()) {
			return -1;
		} else if(c1.getAverageError() > c2.getAverageError()) {
			return 1;
		} else {
			return 0;
		}
	}

}
