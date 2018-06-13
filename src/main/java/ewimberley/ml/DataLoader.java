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
				String line = csvScanner.nextLine();
				lines.add(line);
			}
			csvScanner.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

		int numFeatures = lines.get(0).split(",").length - 1;
		classLabels = new String[lines.size()];
		
		List<double[]> dataList = new ArrayList<double[]>();

		for (int i = 0; i < lines.size(); i++) {
			String[] values = lines.get(i).split(",");
			try {
				double[] fields = new double[numFeatures];
				for (int j = 0; (j < values.length - 1); j++) {
					// FIXME add nominal/categorical variables
					fields[j] = Double.parseDouble(values[j]);
				}
				dataList.add(fields);
			} catch (Exception e) {
				//throw this data item out
			}
			data = new double[dataList.size()][numFeatures];
			for(int k = 0; k < dataList.size(); k++) {
				data[k] = dataList.get(k);
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
