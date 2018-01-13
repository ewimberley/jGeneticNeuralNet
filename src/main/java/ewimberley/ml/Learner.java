package ewimberley.ml;

import java.util.Random;
import java.util.Set;

/**
 * The parent class of all classifiers.
 * 
 * @author ewimberley
 *
 */
public abstract class Learner {

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
	 * Get a random number within a range (inclusive).
	 * 
	 * @param min
	 *            the minimum number in the range
	 * @param max
	 *            the maximum nunber in the range
	 * @return a random integer
	 */
	//XXX test this
	public int getRandInt(int min, int max) {
		return rand.nextInt((max - min) + 1) + min;
	}

	public Learner() {
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