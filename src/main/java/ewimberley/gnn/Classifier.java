package ewimberley.gnn;

import java.util.Random;
import java.util.Set;

public abstract class Classifier {

	protected String[] classLabels;
	protected double[][] data;
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

	public double getRandomDouble() {
		return rand.nextDouble();
	}

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
				if(j == 0) {
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

}