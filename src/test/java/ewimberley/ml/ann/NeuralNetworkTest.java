package ewimberley.ml.ann;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;

import org.junit.Before;
import org.junit.Test;

public class NeuralNetworkTest {
	
	private MockNeuralNetwork network;

	@Before
	public void setup() {
		network = new MockNeuralNetwork();
	}
	
	@Test
	public void testAddInput() {
		InputNeuron in = new InputNeuron(network);
		network.addInput(in);
		assertEquals(1, network.getInputs().size());
		assertSame(in, network.getInputs().toArray(new InputNeuron[] {})[0]);
		assertSame(in, network.getNeurons().get(in.getUuid()));
	}

}
