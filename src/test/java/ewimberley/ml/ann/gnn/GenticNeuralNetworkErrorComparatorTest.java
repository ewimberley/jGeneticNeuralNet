package ewimberley.ml.ann.gnn;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import ewimberley.ml.ann.gnn.classifier.ClassificationGenticNeuralNetwork;

public class GenticNeuralNetworkErrorComparatorTest {
	
	@Test
	public void testCompareSame() {
		GenticNeuralNetwork<String> cnn1 = new ClassificationGenticNeuralNetwork(null, null);
		GenticNeuralNetworkErrorComparator comp = new GenticNeuralNetworkErrorComparator();
		assertEquals(0, comp.compare(cnn1, cnn1));
	}
	
	@Test
	public void testCompareOneGreater() {
		GenticNeuralNetwork<String> cnn1 = new ClassificationGenticNeuralNetwork(null, null);
		cnn1.setAverageError(1.0);
		GenticNeuralNetwork<String> cnn2 = new ClassificationGenticNeuralNetwork(null, null);
		cnn2.setAverageError(0.5);
		GenticNeuralNetworkErrorComparator comp = new GenticNeuralNetworkErrorComparator();
		assertEquals(1, comp.compare(cnn1, cnn2));
	}
	
	@Test
	public void testCompareTwoGreater() {
		GenticNeuralNetwork<String> cnn1 = new ClassificationGenticNeuralNetwork(null, null);
		cnn1.setAverageError(0.5);
		GenticNeuralNetwork<String> cnn2 = new ClassificationGenticNeuralNetwork(null, null);
		cnn2.setAverageError(1.0);
		GenticNeuralNetworkErrorComparator comp = new GenticNeuralNetworkErrorComparator();
		assertEquals(-1, comp.compare(cnn1, cnn2));
	}

}
