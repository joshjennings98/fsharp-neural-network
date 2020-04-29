// Future improvements:
//
// Add more optimisers
// Add more initialisers
// Add graphs
// Add datasets
// Add more loss functions
// Add more regularisers
// Add different types of layer (conv and pooling)
// Add more options for hyperparameters such as clip values etc.

open Types
open NeuralNetwork

[<EntryPoint>]
let main argv =
    
    // Define the network architecture
    let networkArchitecture = [
        {inputDims = 3; outputDims = 5; activation = Sigmoid};
        {inputDims = 5; outputDims = 6; activation = Sigmoid};
        {inputDims = 6; outputDims = 2; activation = Sigmoid};
    ]

    // Make some data
    let data = [
            [0.5; 1.0; 0.2];
            [0.1; 0.7; 1.0];
            [1.0; 0.1; 0.1];
            [0.0; 0.34; 0.8];
            [0.6; 0.1; 0.3]
        ]

    // Make labels for the data
    let labels = [
            [1.0; 1.0];
            [0.0; 1.0];
            [0.0; 0.0];
            [1.0; 0.0];
            [0.0; 1.0];
        ]

    // Train the model using the labels and data
    let model = trainNetwork networkArchitecture labels data 0.05 MSE 100000

    // See the output matches up
    for idx in List.init (List.length data) id do
        printfn "Input: %A" data.[idx]
        printfn "Output: %A" (runNetwork model data.[idx] networkArchitecture)
        printfn "Actual: %A\n" labels.[idx]

    // Try a new input
    printfn "Input: %A" [0.1; 0.8; 0.4]
    printfn "Output: %A" (runNetwork model [0.1; 0.8; 0.4] networkArchitecture)

    0 // return code
