package ewimberley.ml;

import java.util.Random;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

/**
 * The parent class of all classifiers.
 * 
 * @author ewimberley
 *
 */
public abstract class Learner<Y> {

	private Y[] y;
	
	private double[][] data;
	
	protected Set<String> uniqueClassLabels;
	
	public Learner(double[][] data, Y[] y) {
		this.setData(data);
		this.setY(y);
	}

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
		//return rand.nextDouble();
		return ThreadLocalRandom.current().nextDouble();
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
	public int getRandInt(int min, int max) {
		//return rand.nextInt((max - min) + 1) + min;
		return ThreadLocalRandom.current().nextInt((max - min) + 1) + min;
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

	public Y[] getY() {
		return y;
	}

	public void setY(Y[] y) {
		this.y = y;
	}

}