// To Do:
//
// Add a function that lets you do (int * Activation) list -> Layer list
// Make it so catergorical cross entropy doesn't break if there are too many iterations

open System
open Activations
open Losses
open Types
open NeuralNetwork

[<EntryPoint>]
let main argv =
    
    let networkArchitecture = [
        {inputDims = 2; outputDims = 2; activation = Sigmoid};
        {inputDims = 2; outputDims = 2; activation = Sigmoid};
    ]

    let networkArchitecture = [
        {inputDims = 3; outputDims = 3; activation = Relu};
        {inputDims = 3; outputDims = 3; activation = Relu};
        {inputDims = 3; outputDims = 3; activation = Sigmoid};
    ]

    let x = trainNetwork networkArchitecture [0.0; 1.0; 0.0] [0.1; 0.2; 0.7] 0.005 MAE 10000
    
    let output, error = runNetwork x [0.1; 0.2; 0.7] networkArchitecture CategoricalCrossEntropy [0.0; 1.0; 0.0]

    let nicePrint = "\nOutput: " + (sprintf "%A" output) + "\nError: " + (error |> string) + "\n\n" // sprintf should convert the whole list to string if its very long
    
    printf "%A" nicePrint

    0 // return an integer exit code
