// To Do:
//
// Add optimisers
// Add initialisers
// Add graphs
// Add datasets
// Add more loss functions
// Add regularisers
// Add different types of layer (conv and pooling)
// Add more options for hyperparameters such as clip values etc.
// Fix nan problem

open System
open Activations
open Losses
open Types
open NeuralNetwork

[<EntryPoint>]
let main argv =

    let networkArchitecture = [
        {inputDims = 3; outputDims = 3; activation = Relu};
        {inputDims = 3; outputDims = 3; activation = Sigmoid};
        {inputDims = 3; outputDims = 3; activation = Softmax};
    ]

    let x = trainNetwork networkArchitecture [0.0; 1.0; 0.0] [0.1; 0.2; 0.7] 0.005 CrossEntropy 10000
    
    let output, error = runNetwork x [0.1; 0.2; 0.7] networkArchitecture CrossEntropy [0.0; 1.0; 0.0]

    let nicePrint = "\nOutput: " + (sprintf "%A" output) + "\nError: " + (error |> string) + "\n\n" // sprintf should convert the whole list to string if its very long
    
    printf "%A" nicePrint

    0 // return an integer exit code
