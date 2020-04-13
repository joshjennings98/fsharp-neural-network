# F# Neural Network Maker

A scalable fully-connected neural network maker developed for and built using FSharp.

## Usage

Networks can be specified using the following format. Any number of layers are supported.

```fsharp
let networkArchitecture = [
        {inputDims = 3; outputDims = 3; activation = Relu};
        {inputDims = 3; outputDims = 3; activation = Sigmoid};
        {inputDims = 3; outputDims = 3; activation = Softmax};
    ]
```

To train and run the network, see the code snippets below:

```fsharp
    let x = trainNetwork networkArchitecture [0.0; 1.0; 0.0] [0.1; 0.2; 0.7] 0.005 CrossEntropy 10000
    
    let output, error = runNetwork x [0.1; 0.2; 0.7] networkArchitecture CrossEntropy [0.0; 1.0; 0.0]
```

## Features

Choose from the following activation functions:
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

Choose from the following loss functions:
* Mean Square Error
* Cross Entropy 
* Mean Absolute Error

## Bugs

The networks can die easily if the chosen parameters cause weigths to overflow and become NaN.