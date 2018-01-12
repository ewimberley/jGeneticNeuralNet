package ewimberley.ml;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class DataLoader {
	
	private String[] classLabels;
	private double[][] data;

	public void loadCSVFile(String dataFile) {
		File file = new File(dataFile);
		Scanner csvScanner = null;
		List<String> lines = new ArrayList<String>();
		try {
			csvScanner = new Scanner(file);
			while (csvScanner.hasNext()) {
				String line = csvScanner.next();
				lines.add(line);
			}
			csvScanner.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

		int numFeatures = lines.get(0).split(",").length - 1;
		classLabels = new String[lines.size()];
		data = new double[lines.size()][numFeatures];

		for (int i = 0; i < lines.size(); i++) {
			String[] values = lines.get(i).split(",");
			for (int j = 0; (j < values.length - 1); j++) {
				data[i][j] = Double.parseDouble(values[j]);
			}
			classLabels[i] = values[values.length - 1];
		}
	}

	public String[] getClassLabels() {
		return classLabels;
	}

	public double[][] getData() {
		return data;
	}
	
}
