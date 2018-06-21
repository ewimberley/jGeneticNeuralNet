package ewimberley.ml.ann;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import org.junit.Test;

import ewimberley.ml.ann.gnn.ContinuousHiddenNeuron;
import ewimberley.ml.ann.gnn.GenticNeuralNetwork;
import ewimberley.ml.ann.gnn.classifier.ClassificationGenticNeuralNetwork;

public class ContinuousHiddenNeuronTest {

	@Test
	public void testActivationSum() {
		GenticNeuralNetwork<String> network = new ClassificationGenticNeuralNetwork(null, null);
		InputNeuron n1 = new InputNeuron(network);
		n1.setInput(1.0);
		network.addInput(n1);
		ContinuousHiddenNeuron n2 = new ContinuousHiddenNeuron(network);
		network.neurons.put(n2.getUuid(), n2);
		n1.next.add(n2.getUuid());
		n1.nextWeights.put(n2.getUuid(), 1.0);
		n2.prev.add(n1.getUuid());
		assertEquals(0.75, n2.activation(), 0.01);
		n2.resetMemoization();
		n1.setInput(0.0);
		assertEquals(0.5, n2.activation(), 0.01);
	}

	@Test
	public void testActivationBias() {
		ContinuousHiddenNeuron n = new ContinuousHiddenNeuron(null);
		n.setBias(0.0);
		assertEquals(0.5, n.activation(), 0.01);

		n.resetMemoization();
		n.setBias(0.5);
		assertEquals(0.6475, n.activation(), 0.01);

		n.resetMemoization();
		n.setBias(1.0);
		assertEquals(0.75, n.activation(), 0.01);
	}

	@Test
	public void testActivationFunction() {
		ContinuousHiddenNeuron n = new ContinuousHiddenNeuron(null);
		n.setBias(1.0);
		assertEquals(0.75, n.activation(), 0.01);

		n.resetMemoization();
		n.setActivationFunction(ActivationFunction.SIN);
		assertEquals(1.0, n.activation(), 0.01);

		n.resetMemoization();
		n.setActivationFunction(ActivationFunction.STEP);
		assertEquals(1.0, n.activation(), 0.01);
		
		n.resetMemoization();
		n.setActivationFunction(ActivationFunction.LINEAR);
		assertEquals(1.0, n.activation(), 0.01);
	}

	@Test
	public void testSinusoidal() {
		ContinuousHiddenNeuron n = new ContinuousHiddenNeuron(null);
		assertEquals(0.0, n.sinusoidal(0.0), 0.001);
		assertEquals(0.146, n.sinusoidal(0.25), 0.001);
		assertEquals(0.5, n.sinusoidal(0.5), 0.001);
		assertEquals(0.853, n.sinusoidal(0.75), 0.001);
		assertEquals(1.0, n.sinusoidal(1.0), 0.001);
	}

	@Test
	public void testUUID() {
		ContinuousHiddenNeuron n1 = new ContinuousHiddenNeuron(null);
		ContinuousHiddenNeuron n2 = new ContinuousHiddenNeuron(null);
		assertNotEquals(n1.getUuid(), n2.getUuid());
	}

}
