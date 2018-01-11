package ewimberley.gnn;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import org.junit.Test;

public class NeuronTest {

	@Test
	public void testSinusoidal() {
		Neuron n = new HiddenNeuron(null);
		assertEquals(0.0, n.sinusoidal(0.0), 0.001);
		assertEquals(0.146, n.sinusoidal(0.25), 0.001);
		assertEquals(0.5, n.sinusoidal(0.5), 0.001);
		assertEquals(0.853, n.sinusoidal(0.75), 0.001);
		assertEquals(1.0, n.sinusoidal(1.0), 0.001);
	}
	
	@Test
	public void testUUID() {
		Neuron n1 = new HiddenNeuron(null);
		Neuron n2 = new HiddenNeuron(null);
		assertNotEquals(n1.getUuid(), n2.getUuid());
	}

}
