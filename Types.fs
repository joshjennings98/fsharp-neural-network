module Types


type Layer = {inputDims : int; outputDims : int; activation : Activation}


type Parameters = {weights : float list list list; biases : float list}


type Activation =
    | Relu
    | Sigmoid
    | Tanh
    | Softmax
    | LeakyRelu of float
    | Elu of float
    | Selu of float * float
    | Softsign
    | Softplus
    | Exponential
    | HardSigmoid
    | Linear


type Loss =
    | MSE
    | CrossEntropy 
    | MAE