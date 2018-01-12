package ewimberley.ml;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class ConfusionMatrix {
	
	private int[][] confusionMatrix;
	private Set<String> uniqueClassLabels;
	private Map<String, Integer> classLabelConfusionMatrixIndices;
	String[] confusionMatrixIndicesToClassLabel;
	
	public ConfusionMatrix(String[] classLabels) {
		uniqueClassLabels = calculateUniqueClassLabels(classLabels);
		classLabelConfusionMatrixIndices = new HashMap<String, Integer>();
		confusionMatrixIndicesToClassLabel = new String[uniqueClassLabels.size()];
		int onConfusionMatrixIndex = 0;
		for (String uniqueClassLabel : uniqueClassLabels) {
			classLabelConfusionMatrixIndices.put(uniqueClassLabel, onConfusionMatrixIndex);
			confusionMatrixIndicesToClassLabel[onConfusionMatrixIndex] = uniqueClassLabel;
			onConfusionMatrixIndex++;
		}
		// first dimension is expected, second dimension is predicted
		confusionMatrix = new int[uniqueClassLabels.size()][uniqueClassLabels.size()];
	}
	
	public void printConfusionMatix() {
		int numTestingSamples = 0;
		for (int i = 0; i < confusionMatrix.length; i++) {
			if (i == 0) {
				System.out.print("expected/predicted\t");
				for (int j = 0; j < confusionMatrix[i].length; j++) {
					System.out.print(confusionMatrixIndicesToClassLabel[j] + "\t");
				}
				System.out.println();
			}
			for (int j = 0; j < confusionMatrix[i].length; j++) {
				if (j == 0) {
					System.out.print(confusionMatrixIndicesToClassLabel[i] + "\t\t");
				}
				System.out.print(confusionMatrix[i][j] + "\t");
				numTestingSamples += confusionMatrix[i][j];
			}
			System.out.println();
		}

		int numCorrect = 0;
		for (int i = 0; i < confusionMatrix.length; i++) {
			numCorrect += confusionMatrix[i][i];
		}
		double accuracy = (double) numCorrect / (double) numTestingSamples;
		System.out.println("Accuracy: " + accuracy);
	}
	
	private static HashSet<String> calculateUniqueClassLabels(String[] classLabels) {
		HashSet<String> uniqueClassLabels = new HashSet<String>();
		for (String classLabel : classLabels) {
			uniqueClassLabels.add(classLabel);
		}
		return uniqueClassLabels;
	}

	public int[][] getConfusionMatrix() {
		return confusionMatrix;
	}

	public void setConfusionMatrix(int[][] confusionMatrix) {
		this.confusionMatrix = confusionMatrix;
	}

	public Map<String, Integer> getClassLabelConfusionMatrixIndices() {
		return classLabelConfusionMatrixIndices;
	}

	public void setClassLabelConfusionMatrixIndices(Map<String, Integer> classLabelConfusionMatrixIndices) {
		this.classLabelConfusionMatrixIndices = classLabelConfusionMatrixIndices;
	}

}
