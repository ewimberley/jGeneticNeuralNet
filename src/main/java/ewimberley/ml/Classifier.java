package ewimberley.ml;

import java.util.Random;
import java.util.Set;

/**
 * The parent class of all classifiers.
 * 
 * @author ewimberley
 *
 */
public abstract class Classifier {

	private String[] classLabels;
	private double[][] data;
	protected Set<String> uniqueClassLabels;
	protected Random rand;

	public static void printTrainingExample(double[] features, String label) {
		for (int i = 0; i < features.length; i++) {
			if (i > 0) {
				System.out.print(", ");
			}
			System.out.print(features[i]);
		}
		System.out.println(", " + label);
	}

	/**
	 * Get a random number between 0 and 1.
	 * 
	 * @return a random double
	 */
	public double getRandomDouble() {
		return rand.nextDouble();
	}

	/**
	 * Get a random number within a range.
	 * 
	 * @param min
	 *            the minimum number in the range
	 * @param max
	 *            the maximum nunber in the range
	 * @return a random integer
	 */
	public int getNextInt(int min, int max) {
		return rand.nextInt((max - min) + 1) + min;
	}

	protected static void printConfusionMatix(String[] confusionMatrixIndicesToClassLabel, int[][] confusionMatrix) {
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

	public Classifier() {
		super();
	}

	public double[][] getData() {
		return data;
	}

	public void setData(double[][] data) {
		this.data = data;
	}

	public String[] getClassLabels() {
		return classLabels;
	}

	public void setClassLabels(String[] classLabels) {
		this.classLabels = classLabels;
	}

}