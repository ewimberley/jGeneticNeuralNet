package ewimberley.ml;

/**
 * A configuration object for training machine learning models.
 * 
 * @author ewimberley
 *
 */
public class ModelTrainingConfig {
	
	//FIXME add cross-fold validation configuration items here
	//FIXME add ROC and prec/recall curve config here

	private int maxThreads;

	public ModelTrainingConfig() {
		maxThreads = Runtime.getRuntime().availableProcessors() * 2;
	}

	public int getMaxThreads() {
		return maxThreads;
	}

	public void setMaxThreads(int maxThreads) {
		if (maxThreads < 1) {
			throw new IllegalArgumentException("Number of threads must be at least 1.");
		}
		this.maxThreads = maxThreads;
	}

}