// To Do:
//
// Add a function that lets you do (int * Activation) list -> Layer list

open System
open NeuralNetwork

[<EntryPoint>]
let main argv =
    
    let networkArchitecture = [
        {inputDims = 2; outputDims = 2; activation = Sigmoid};
        {inputDims = 2; outputDims = 2; activation = Sigmoid};
    ]

    let networkArchitecture = [
        {inputDims = 3; outputDims = 5; activation = Relu};
        {inputDims = 5; outputDims = 10; activation = Relu};
        {inputDims = 10; outputDims = 7; activation = Relu};
        {inputDims = 7; outputDims = 4; activation = Sigmoid};
    ]

    let x = trainNetwork networkArchitecture [0.01; 0.99; 0.35; 0.4] [0.05; 0.1; 0.5] 0.05 MSE 10000
    
    let output, error = runNetwork x [0.05; 0.1; 0.5] networkArchitecture MSE [0.01; 0.99; 0.35; 0.4]

    let nicePrint = "\nOutput: " + (sprintf "%A" output) + "\nError: " + (error |> string) + "\n\n" // sprintf should convert the whole list to string if its very long
    
    printf "%A" nicePrint
    
    0 // return an integer exit code
