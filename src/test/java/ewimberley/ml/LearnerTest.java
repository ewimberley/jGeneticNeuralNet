package ewimberley.ml;

import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;

public class LearnerTest {

	private MockLearner learn;

	@Before
	public void setup() {
		learn = new MockLearner();
	}

	@Test
	public void testGetRandInt() {
		for(int i = 0; i < 10; i++) {
			int next = learn.getRandInt(0, 9);
			assertTrue(next <= 9);
			assertTrue(next >= 0);
		}
	}
	
	@Test
	public void testGetRandomDouble() {
		for(int i = 0; i < 10; i++) {
			double next = learn.getRandomDouble();
			assertTrue(next <= 1.0);
			assertTrue(next >= 0.0);
		}
	}

}
