package ewimberley.ml.ann.gnn;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import org.junit.Test;

import ewimberley.ml.ann.gnn.ContinuousHiddenNeuron;

public class ContinuousHiddenNeuronTest {

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
