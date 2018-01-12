package ewimberley.ml;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import ewimberley.ml.DataLoader;

public class DataLoaderTest {

	@Test
	public void testIrisData() {
		String dataFile = "src/test/resources/iris.data";
		DataLoader dl = new DataLoader();
		dl.loadCSVFile(dataFile);
		assertEquals(150, dl.getData().length);
		assertEquals(4, dl.getData()[0].length);
		assertEquals(150, dl.getClassLabels().length);
	}

}
