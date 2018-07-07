# jGeneticNeuralNet
A Java library that implements neural networks with a genetic training algorithm.

## Classification Usage

```java
String dataFile = "src/test/resources/iris.data";
DataLoader dl = new DataLoader();
dl.loadCSVFile(dataFile);
NeuralNetworkTrainingConfiguration config = new NeuralNetworkTrainingConfiguration();
config.setNumNetworksPerGeneration(3000);
config.setNumGenerations(500);
config.setNumHiddenLayers(3);
config.setNumNeuronsPerLayer(8);
config.setMaxLearningRate(10.0);
config.setVisualizer(vis);
GenticNeuralNetwork<String> model = ClassificationGenticNeuralNetwork.train(dl.getData(),  dl.getClassLabels(), config);
```

```
Expected Iris-setosa, predicted Iris-setosa with probability 0.9960185747628352
Expected Iris-setosa, predicted Iris-setosa with probability 0.9960185582287887
...
Expected Iris-versicolor, predicted Iris-virginica with probability 0.9955454129025534
Expected Iris-versicolor, predicted Iris-versicolor with probability 0.9894226988148483
expected/predicted	Iris-versicolor	Iris-virginica	Iris-setosa	
Iris-versicolor		9	1	0	
Iris-virginica		0	10	0	
Iris-setosa		0	0	11	
Accuracy: 0.967741935483871
```

## Regression Usage

```java
NeuralNetworkTrainingConfiguration config = new NeuralNetworkTrainingConfiguration();
config.setNumNetworksPerGeneration(1000);
config.setNumGenerations(500);
config.setNumHiddenLayers(2);
config.setNumNeuronsPerLayer(6);
config.setMaxLearningRate(1.0);
config.setVisualizer(vis);
config.setProbMutateActivationFunction(0.1);
RegressionGenticNeuralNetwork bestNetwork = RegressionGenticNeuralNetwork.train(data, values, config);
```
