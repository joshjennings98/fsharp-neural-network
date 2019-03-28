#include <iostream>
#include <vector>
#include <utility>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include <chrono>

#include "MNIST.h"

struct network {
  std::vector<std::vector<double> > weights;
  std::vector<std::vector<double> > biases;
};

struct data {
  std::vector<double> input;
  std::vector<double> output;
  int label;
};

typedef std::vector<std::pair<int, std::string> > defineNetwork;

const int NUMBER_OF_ITERATIONS = 5000;
const double LEARNING_RATE = 0.005;

double randomValue(double mean, double standardDeviation);
double setValue();
double sigmoid(double x);
double dSigmoid(double x);
double relu(double x);
double dRelu(double x);
double activation(double x, std::string activationType);
double dActivation(double x, std::string activationType);
void initialiseLayers(std::vector<std::pair<int, std::string> > neuralArchitecture, std::vector<std::vector<double> >& weights, std::vector<std::vector<double> >& biases, std::vector<double> inputs);
void forwardFull(std::vector<std::pair<int, std::string> > neuralArchitecture, std::vector<std::vector<double> >& weights, std::vector<std::vector<double> >& biases, std::vector<double>& inputs, std::vector<std::vector<double> >& memory, std::vector<std::vector<double> >& memoryActivated);
void forwardSinglelayer(std::vector<double>& biases, std::vector<double>& weights, std::vector<double>& inputs, std::vector<double>& tempWeights, std::vector<double>& tempweightsInactivated, std::string activationType);
void getErrors(std::vector<double> networkOutput, std::vector<double> testOutput, std::vector<double>& errors, double& errorTotal);
void backpropogation(std::vector<std::pair<int, std::string> > neuralArchitecture, std::vector<std::vector<double> >& activated, std::vector<std::vector<double> >& weights, std::vector<std::vector<double> >& tempWeights, std::vector<double> networkOutput, std::vector<double> testOutput, std::vector<double> testInput, std::vector<std::vector<double> >& partials, double learningRate, int iteration);
void buildNetwork(std::vector<data>& data, std::vector<std::pair<int, std::string> >& architecture, std::string fileName, std::vector<std::pair<int, double> >& allErrors);
void loadNetwork(network& network, std::string fileName);
void runNetwork(network network, std::vector<double> input, std::vector<double>& actualOutput, defineNetwork neuralArchitecture);
//Need to sort out learning rate, add leaky relu and more, gradient clipping, bias updater, turn into header file.

int main() {
std::vector<std::pair<int, double> > allErrors;

data dataVal;
std::vector<data> dataSet;
std::vector<std::vector<double> > tempInputs;
std::vector<std::vector<double> > tempOutputs;
std::vector<int> labels;

MNIST mnist = MNIST("/run/media/josh/Tiny USB/");
mnist.loadVectors(0, 20, tempInputs, tempOutputs, labels);
for (int i = 0; i < tempInputs.size(); i++) {
  dataVal.input = tempInputs.at(i);
  dataVal.output = tempOutputs.at(i);
  dataVal.label = labels.at(i);
  dataSet.push_back(dataVal);
}
//dataSet.clear()
 /*
dataVal.input = {0.0, 0.0, 1.0};
dataVal.output = {0.0,1.0};
dataSet.push_back(dataVal);
dataVal.input = {0.0, 1.0, 1.0};
dataVal.output = {1.0,1.0};
dataSet.push_back(dataVal);
// */

network newNetwork;
std::vector<double> actualOutput;
std::string fileName = "networktest.txt";
//defineNetwork neuralArchitecture = {{2, "relu"}, {4, "relu"}, {2, "sigmoid"}};
//defineNetwork neuralArchitecture = {{4, "relu"}, {6, "relu"}, {6, "relu"}, {4, "relu"}, {2, "sigmoid"}};
defineNetwork neuralArchitecture = {{32, "relu"}, {10, "sigmoid"}};


buildNetwork(dataSet, neuralArchitecture, fileName, allErrors);
loadNetwork(newNetwork, fileName);

for (int i = 0; i < dataSet.size(); i++) {
  runNetwork(newNetwork, dataSet.at(i).input, actualOutput, neuralArchitecture);
  //for (int j = 0; j < dataSet.at(i).input.size(); j++) {
  //  std::cout << dataSet.at(i).input.at(j) << " ";
  //}
  std::cout << dataSet.at(i).label << " = ";
  for (int j = 0; j < actualOutput.size(); j++) {
    std::cout << actualOutput.at(j) << " ";
  }
  actualOutput.clear();
  std::cout << std::endl;
}

/*
std::cout << "plot { ";
for (int i = 0; i < allErrors.size(); i++) {
	std::cout << "{ " << allErrors.at(i).first << " ," << allErrors.at(i).second << " } ";
}
std::cout << "}" << std::endl;
*/

return 0;
}

double randomValue(double mean, double standardDeviation) {
  double r = -1;
  std::random_device dre;
  std::default_random_engine generator(dre());
  std::normal_distribution<double> distribution(mean, standardDeviation);
  while(r <= 0 or r >= 10) {
  r = distribution(generator);
}
  return r * 0.1;
}

double setValue() {
  double r = 0.0;
  return r * 0.01;
}

double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-1.0 * x));
}

double dSigmoid(double x) {
  return x * (1 - x);
}

double relu(double x) {
  if (x >= 0) {
    return x;
  }
  else {
    return 0.0;
  }
}

double dRelu(double x){
  if (x >= 0) {
    return 1.0;
  }
  else {
    return 0.0;
  }
}

void initialiseLayers(std::vector<std::pair<int, std::string> > neuralArchitecture, std::vector<std::vector<double> >& weights, std::vector<std::vector<double> >& biases, std::vector<double> inputs) {
  int numLayers = neuralArchitecture.size();
  int layerInputsize;
  int layerOutputsize;
  double standardDeviation;
  std::vector<double> tempWeights;
  std::vector<double> tempBiases;

  for (int i = 0; i < numLayers; i++) {
    if (i != 0) {
      layerInputsize = neuralArchitecture.at(i - 1).first;
      layerOutputsize = neuralArchitecture.at(i).first;
    }
    else {
      layerInputsize = inputs.size();
      layerOutputsize = neuralArchitecture.at(i).first;
    }
    tempWeights.resize(layerInputsize * layerOutputsize);
    tempBiases.resize(layerOutputsize);
    standardDeviation = sqrt(2 / ((double) layerInputsize + (double) layerOutputsize));
    for (int j = 0; j < tempWeights.size(); j++) {
      tempWeights.at(j) = randomValue(0.0, standardDeviation);
    }
    std::fill(tempBiases.begin(), tempBiases.end(), 1);
    weights.push_back(tempWeights);
    biases.push_back(tempBiases);
  }
}

void forwardSinglelayer(std::vector<double>& biases, std::vector<double>& weights, std::vector<double>& inputs, std::vector<double>& tempWeights, std::vector<double>& tempweightsInactivated, std::string activationType) {
  double temp = 0;
  int count = 0;

  for (int i = 0; i < biases.size(); i++) {
    for (int j = 0; j < inputs.size(); j++) {
      temp = temp + weights.at(count) * inputs.at(j);
      count++;
    }
    tempWeights.push_back(activation(temp + biases.at(i), activationType));
    tempweightsInactivated.push_back(temp + biases.at(i));
    temp = 0;
  }
}

double activation(double x, std::string activationType) {
  if (activationType == "sigmoid") {
    return sigmoid(x);
 }
  else if (activationType == "relu") {
    return relu(x);
  } else {
    std::cout << "Invalid Activation Function." << std::endl;
    std::exit(0);
  }
}

double dActivation(double x, std::string activationType) {
  if (activationType == "sigmoid") {
    return dSigmoid(x);
 }
  else if (activationType == "relu") {
    return dRelu(x);
  } else {
    std::cout << "Invalid Activation Function." << std::endl;
    std::exit(0);
  }
}

void forwardFull(std::vector<std::pair<int, std::string> > neuralArchitecture, std::vector<std::vector<double> >& weights, std::vector<std::vector<double> >& biases, std::vector<double>& inputs, std::vector<std::vector<double> >& memory, std::vector<std::vector<double> >& memoryActivated) {
  int numLayers = neuralArchitecture.size();
  std::vector<double> tempOutputs;
  std::vector<double> tempoutputsInactivated;

  memory.clear();
  memoryActivated.clear();
  forwardSinglelayer(biases.at(0), weights.at(0), inputs, tempOutputs, tempoutputsInactivated, neuralArchitecture.at(0).second);
  memoryActivated.push_back(tempOutputs);
  memory.push_back(tempoutputsInactivated);
  for (int i = 0; i < numLayers - 1; i++) {
    tempOutputs.clear();
    tempoutputsInactivated.clear();
    forwardSinglelayer(biases.at(i + 1), weights.at(i + 1), memoryActivated.at(i), tempOutputs, tempoutputsInactivated, neuralArchitecture.at(i + 1).second);
    memoryActivated.push_back(tempOutputs);
    memory.push_back(tempoutputsInactivated);
  }
}

void getErrors(std::vector<double> networkOutput, std::vector<double> testOutput, std::vector<double>& errors, double& errorTotal) {
  errors.clear();
  errorTotal = 0;
  for (int i = 0; i < testOutput.size(); i++){
  	errors.push_back(testOutput.at(i) - networkOutput.at(i));
  }
  for (int i = 0; i < errors.size(); i++) {
    errorTotal = errorTotal + 0.5 * pow(errors.at(i), 2);
  }
}

void backpropogation (std::vector<std::pair<int, std::string> > neuralArchitecture, std::vector<std::vector<double> >& activated, std::vector<std::vector<double> >& weights, std::vector<std::vector<double> >& tempWeights, std::vector<double> networkOutput, std::vector<double> testOutput, std::vector<double> testInput, std::vector<std::vector<double> >& partials, double learningRate, int iteration) {
  double temp1 = 0;
	double tempPartial = 0;
	std::vector<double> tempPartials = {};
  std::vector<double> tempPartials2 = {};
	int count = 0;
  std::string activationType;
  double totalPartial;

	for (int i = 0; i < tempWeights.size() - 1; i++) { //each layer
    activationType = neuralArchitecture.at(i).second;
		for (int j = 0; j < activated.at(activated.size() - 1 - i).size(); j++) {
			for (int k = 0; k < activated.at(activated.size() - 2 - i).size(); k++) {
				temp1 = dActivation(activated.at(activated.size() - 1 - i).at(j), activationType) * (networkOutput.at(j % networkOutput.size()) - testOutput.at(j % testOutput.size())); //Issues caused by this line???? Are they fixed????
				if (i > 0) {
					tempPartial = tempPartial * partials.at(partials.size() - 1).at(k % partials.at(partials.size() - 1).size()); //Issues caused by this line???? Are they fixed????
				}
        tempPartials.push_back(temp1 * weights.at(weights.size() - 1 - i).at(count));
        temp1 = temp1 * activated.at(activated.size() - 2 - i).at(k);
				tempWeights.at(tempWeights.size() - 1 - i).at(count) = tempWeights.at(tempWeights.size() - 1 - i).at(count) - learningRate * temp1;
				temp1 = 0;
				count++;
			}
		}
    totalPartial = 0;
    for (int j = 0; j < activated.at(activated.size() - 1 - i).size(); j++) {
      for (int k = 0; k < activated.at(activated.size() - 2 - i).size(); k++) {
        totalPartial = totalPartial + tempPartials.at(j + k * activated.at(activated.size() - 1 - i).size());
      }
      tempPartials2.push_back(totalPartial);
      totalPartial = 0;
    }
		partials.push_back(tempPartials2);
    tempPartials.clear();
		count = 0;
	}
	count = 0;
	for (int j = 0; j < activated.at(activated.size() - 1).size(); j++) {
			for (int k = 0; k < testInput.size(); k++) {
			  tempPartial = dActivation(activated.at(0).at(j), activationType) * testInput.at(k);
				tempPartial = tempPartial * partials.at(partials.size() - 1).at(j);
        temp1 = tempPartial;
				tempWeights.at(0).at(count) = tempWeights.at(0).at(count) - learningRate * temp1;
				if (iteration % 100 == 0 or iteration == 1) {
					//std::cout << tempPartial << " ";
				}
				temp1 = 0;
				count++;
			}
		}
	//count = 0;
}

void buildNetwork (std::vector<data>& data, std::vector<std::pair<int, std::string> >& architecture, std::string fileName, std::vector<std::pair<int, double> >& allErrors) {
  std::ofstream networkCharacteristics;
  std::vector<double> errors;
  int numIterations = NUMBER_OF_ITERATIONS;
  int testValue = 0;
  double errorTotal = 0;
  std::vector<std::vector<double> > weights;
  std::vector<std::vector<double> > biases;
  std::vector<std::vector<double> > memory;
  std::vector<std::vector<double> > memoryActivated;
  std::vector<std::vector<double> > tempWeights;
  std::vector<std::vector<double> > tempBiases;
  std::vector<std::vector<double> > partials;
  std::pair<int, double> tempPair;

  initialiseLayers(architecture, weights, biases, data.at(testValue).input);
  for (int i = 0; i < weights.size(); i++) {
    tempWeights.push_back(weights.at(i));
  }
  forwardFull(architecture, weights, biases, data.at(testValue).input, memory, memoryActivated);
  getErrors(memoryActivated.at(memoryActivated.size() - 1), data.at(testValue).output, errors, errorTotal);
  for (int x = 1; x <= numIterations; x++) {
    testValue = rand() % data.size();
    forwardFull(architecture, weights, biases, data.at(testValue).input, memory, memoryActivated);
    partials.clear();
    backpropogation(architecture, memoryActivated, weights, tempWeights, memoryActivated.at(memoryActivated.size() - 1), data.at(testValue).output, data.at(testValue).input, partials, LEARNING_RATE, x);
    weights.clear();
    for (int i = 0; i < tempWeights.size(); i++) {
      weights.push_back(tempWeights.at(i));
    }
    getErrors(memoryActivated.at(memoryActivated.size() - 1), data.at(testValue).output, errors, errorTotal);
    if (x % 1000 == 0 or x == 1) {
    	std::cout << "Iteration: " << x << std::endl;
      /*
    	std::cout << "Output: ";
    	for (int j = 0; j < memoryActivated.at(memoryActivated.size() - 1).size(); j++) {
    		std::cout << memoryActivated.at(memoryActivated.size() - 1).at(j) << " ";
    	}
    	std::cout << std::endl;
    	std::cout << "Error: " <<  errorTotal << std::endl;
    	tempPair.first = x;
    	tempPair.second = errorTotal;
    	allErrors.push_back(tempPair);
    	std::cout << "Weights: " << std::endl;
    	for (int j = 0; j < weights.size(); j++) {
    		for (int k = 0; k < weights.at(j).size(); k++) {
    			std::cout << weights.at(j).at(k) << " ";
    		}
    		std::cout << std::endl;
    	}
    	std::cout << "Biases: " << std::endl;
    	for (int j = 0; j < biases.size(); j++) {
    		for (int k = 0; k < biases.at(j).size(); k++) {
    			std::cout << biases.at(j).at(k) << " ";
    		}
    		std::cout << std::endl;
    	}
    	std::cout << std::endl;
      */
    }
  }
  networkCharacteristics.open (fileName);
  for (int i = 0; i < weights.size(); i++) {
    for (int j = 0; j < weights.at(i).size(); j++) {
      networkCharacteristics << weights.at(i).at(j) << " ";
    }
    networkCharacteristics << std::endl;
  }
  networkCharacteristics << "weights_end " << std::endl;
  for (int i = 0; i < biases.size(); i++) {
    for (int j = 0; j < biases.at(i).size(); j++) {
      networkCharacteristics << biases.at(i).at(j) << " ";
    }
    networkCharacteristics << std::endl;
  }
  networkCharacteristics << "biases_end " << std::endl;
  networkCharacteristics.close();
}

void loadNetwork(network& network, std::string fileName) {
  std::string line;
  std::ifstream myfile (fileName);
  std::vector<double> tempVals;
  std::vector<std::vector<double> > tempVals2;
  std::string word = "";

  if (myfile.is_open()) {
    while ( getline (myfile,line) ) {
      if (line.compare("weights_end ") == 0) {
        network.weights = tempVals2;
        tempVals2.clear();
      }
      else if (line.compare("biases_end ") == 0) {
        network.biases = tempVals2;
        tempVals2.clear();
      }
      else {
        for (auto x : line) {
          if (x == ' ') {
            tempVals.push_back(stod(word));
            word.clear();
          }
          else {
            word = word + x;
          }
        }
      }
      tempVals2.push_back(tempVals);
      tempVals.clear();
    }
    myfile.close();
  }
  else {
    std::cout << "Unable to open file.";
    std::exit(0);
  }
}

void runNetwork(network network, std::vector<double> input, std::vector<double>& actualOutput, defineNetwork neuralArchitecture) {
  int numLayers = neuralArchitecture.size();
  std::vector<double> tempOutputs;
  std::vector<double> tempoutputsInactivated;
  std::vector<std::vector<double> > memoryActivated;
  std::vector<std::vector<double> > memory;
  forwardSinglelayer(network.biases.at(1), network.weights.at(0), input, tempOutputs, tempoutputsInactivated, neuralArchitecture.at(0).second);
  memoryActivated.push_back(tempOutputs);
  memory.push_back(tempoutputsInactivated);
  for (int i = 0; i < numLayers - 1; i++) {
    tempOutputs.clear();
    tempoutputsInactivated.clear();
    forwardSinglelayer(network.biases.at(i + 2), network.weights.at(i + 1), memoryActivated.at(i), tempOutputs, tempoutputsInactivated, neuralArchitecture.at(i + 1).second);
    memoryActivated.push_back(tempOutputs);
    memory.push_back(tempoutputsInactivated);
  }
  for (int j = 0; j < memoryActivated.at(memoryActivated.size() - 1).size(); j++) {
    actualOutput.push_back(memoryActivated.at(memoryActivated.size() - 1).at(j));
  }
}
