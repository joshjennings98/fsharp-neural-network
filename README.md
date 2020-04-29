# F# Neural Network Maker

A simple but scalable fully-connected neural network maker developed for and built using F#.

## Usage

### Defining a network

Networks can be specified using the following format. Any number of layers are supported. Provided they align, the dimensions can be as large as desired.

```fsharp
let networkArchitecture = [
	{inputDims = 3; outputDims = 5; activation = Sigmoid};
	{inputDims = 5; outputDims = 6; activation = Sigmoid};
	{inputDims = 6; outputDims = 2; activation = Sigmoid};
]
```

You can choose from multiple different activation functions including:
* Relu
* Sigmoid
* Tanh
* Softmax
* Leaky Relu
* Elu
* Selu
* Softsign
* Softplus
* Exponential
* Hard Sigmoid
* Linear

### Training a network

Input data is provided in the form of a list of inputs, and a list of labels corresponding to each input.

```fsharp
// Inputs
let data = [
	[0.5; 1.0; 0.2];
	[0.1; 0.7; 1.0];
	[1.0; 0.1; 0.1];
	[0.0; 0.34; 0.8];
	[0.6; 0.1; 0.3]
]

// Labels
let labels = [
	[1.0; 1.0];
	[0.0; 1.0];
	[0.0; 0.0];
	[1.0; 0.0];
	[0.0; 1.0];
]
```

To train and run the network, see the code snippets below:

```fsharp
// trainNetwork architecture labels data learning-rate loss iterations
let model = trainNetwork networkArchitecture labels data 0.05 MSE 100000
```

Currently, the following loss functions are avaliable:
* Mean Square Error
* Cross Entropy 
* Mean Absolute Error


### Running a trained network

A single run of the network can be specified as follows:

```fsharp
// runNetwork model input architecture
runNetwork model [0.1; 0.8; 0.4] networkArchitecture // [0.9064375283; 0.9983475419]
```

We can test multiple inputs by using a loop:

```fsharp
for idx in List.init (List.length data) id do
	printfn "Input: %A" data.[idx]
	printfn "Output: %A" (runNetwork model data.[idx] networkArchitecture)
```


## Bugs

Be careful with the chosen parameters. The networks can die easily if the chosen parameters cause weigths to overflow and become NaN.
