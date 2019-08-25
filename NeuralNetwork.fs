module NeuralNetwork

open Types
open Losses
open ForwardPropogation
open BackPropogation

let initialiseNetwork (architecture : Layer list) : Parameters =
    let genRandomList (size : int) : float list =
        List.init size (fun _ -> 0.01 * (System.Random().Next(0, 100) |> float))  
    let initialWeights =
        [List.init (List.last architecture).outputDims (fun _ -> [1.0])] // This is for the output layer. This makes backpropogation easier
        |> List.append (List.init architecture.Length (fun index ->
            List.init architecture.[index].outputDims (fun _ -> // These three lines just generate a list of random values for each layer
                genRandomList architecture.[index].inputDims)))
    let inititalBiases =
        genRandomList architecture.Length // This doesn't have an extra layer for the output since bias is only used in forward propogation and forward propogation won't reach this point
    {weights = initialWeights; biases = inititalBiases}

let trainNetwork (architecture : Layer list) (targetOutputs : float list) (inputs : float list) (learningRate : float) (loss : Loss) (iterations : int) : Parameters =
    let weightUpdate (parameters : Parameters) (weights : (float list list * float list) list) : Parameters =
        let newWeights =
            weights
            |> List.map fst // Discard intermediate values leaving only the new weights
        {parameters with weights = newWeights}
    let initial = initialiseNetwork architecture
    printf "Network: %A\n\n" initial
    let rec train (parameters : Parameters) (model : Layer list) (maxIterations : int) (iterations : int) =
        match iterations with
        | x when x = maxIterations - 1 -> 
            printf "%s\n" ("Iteration number: " + (iterations + 1 |> string))
            parameters
        | _ -> 
            printf "%s" (if iterations % 100 = 99 || iterations = 0 then "Iteration number: " + (iterations + 1 |> string) + "\n" else "")
            let fullSingle = 
                forwardFull parameters inputs architecture
                |> backPropFull architecture parameters (List.map (fun el -> [el]) targetOutputs) learningRate loss
                |> weightUpdate parameters
            train fullSingle model maxIterations (iterations + 1)
    train initial architecture iterations 0

let runNetwork (trainedNetwork : Parameters) (input : float list) (architecture : Layer list) (loss : Loss) (targetOutputs : float list) : float list * float =
    let output = forwardFull trainedNetwork input architecture
    let error = getOverallError targetOutputs output.[1] loss // Index of 1 since there was 1.0s added to the front
    output.[1], error
