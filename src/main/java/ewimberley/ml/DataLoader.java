package ewimberley.ml;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

public class DataLoader {

	private static final String ALL = "ALL";
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

		Map<Integer, Map<String, Integer>> columnAndValueToOutputColumn = new HashMap<Integer, Map<String, Integer>>();
		Map<Integer, ColumnType> fileColumnToType = new HashMap<Integer, ColumnType>();
		Map<Integer, Set<String>> fileColumnToPossibleValues = new HashMap<Integer, Set<String>>();
		for (int i = 0; i < lines.size(); i++) {
			String[] values = lines.get(i).split(",");
			for (int j = 0; (j < values.length - 1); j++) {
				try {
					Double.parseDouble(values[j]);
					fileColumnToType.putIfAbsent(j, ColumnType.NUMERIC);
				} catch (NumberFormatException nfe) {
					fileColumnToType.putIfAbsent(j, ColumnType.CATEGORICAL);
					if (!fileColumnToPossibleValues.containsKey(j)) {
						fileColumnToPossibleValues.put(j, new HashSet<String>());
					}
					fileColumnToPossibleValues.get(j).add(values[j]);
				}
			}
		}

		int onOutputColumn = 0;
		for (Map.Entry<Integer, ColumnType> columnTypes : fileColumnToType.entrySet()) {
			if (!columnAndValueToOutputColumn.containsKey(columnTypes.getKey())) {
				columnAndValueToOutputColumn.put(columnTypes.getKey(), new HashMap<String, Integer>());
			}
			if (ColumnType.NUMERIC == columnTypes.getValue()) {
				columnAndValueToOutputColumn.get(columnTypes.getKey()).put(ALL, onOutputColumn);
				onOutputColumn++;
			} else {
				Set<String> columnValues = fileColumnToPossibleValues.get(columnTypes.getKey());
				for (String value : columnValues) {
					columnAndValueToOutputColumn.get(columnTypes.getKey()).put(value, onOutputColumn);
					onOutputColumn++;
				}
			}
		}

		for (int i = 0; i < lines.size(); i++) {
			String[] values = lines.get(i).split(",");
			try {
				double[] fields = new double[onOutputColumn];
				for (int j = 0; (j < values.length - 1); j++) {
					if(fileColumnToType.get(j) == ColumnType.NUMERIC) {
						int outputCol = columnAndValueToOutputColumn.get(j).get(ALL);
						fields[outputCol] = Double.parseDouble(values[j]);
					} else {
						// nominal/categorical variable
						int outputCol = columnAndValueToOutputColumn.get(j).get(values[j]);
						fields[outputCol] = 1.0;
					}
				}
				dataList.add(fields);
			} catch (Exception e) {
				// throw this data item out
			}
			data = new double[dataList.size()][numFeatures];
			for (int k = 0; k < dataList.size(); k++) {
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

	private enum ColumnType {
		NUMERIC, CATEGORICAL
	}

}
