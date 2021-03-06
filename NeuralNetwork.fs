﻿module NeuralNetwork


open Types
open Losses
open ForwardPropogation
open BackPropogation


let initialiseNetwork (architecture : Layer list) : Parameters =
    let rand (lo : float) (hi : float) (step : float) : float = // Need the unit so its a function
        System.Random().Next(int (lo / step), int (hi / step))
        |> float
        |> (*) step

    let genRandomList (size : int) : float list =
        List.init size (fun _ -> rand 0.0 1.0 0.01)  
    
    let initialWeights =
        [List.init (List.last architecture).outputDims (fun _ -> [1.0])] // This is for the output layer. This makes backpropogation easier
        |> List.append (List.init architecture.Length (fun index ->
            List.init architecture.[index].outputDims (fun _ -> // These three lines just generate a list of random values for each layer
                genRandomList architecture.[index].inputDims)))
    
    let inititalBiases =
        genRandomList architecture.Length // This doesn't have an extra layer for the output since bias is only used in forward propogation and forward propogation won't reach this point
    
    {weights = initialWeights; biases = inititalBiases}


let trainNetwork (architecture : Layer list) (targetOutputs : float list list) 
    (inputs : float list list) (learningRate : float) (loss : Loss) (iterations : int) : Parameters =
    
    let weightUpdate (parameters : Parameters) (weights : (float list list * float list) list) : Parameters =
        let newWeights =
            List.map fst weights // Discard intermediate values leaving only the new weights
        
        {parameters with weights = newWeights}
    
    let initial = initialiseNetwork architecture

    let targets = List.map (fun el -> [el]) targetOutputs

    printfn "%A" targets
    
    printfn "Network: %A\n" initial
    
    let rec train (parameters : Parameters) (model : Layer list) (maxIterations : int) (iterations : int) =
        match iterations with
        | x when x = maxIterations - 1 -> 
            printfn "Iteration number: %A" (iterations + 1)
            parameters
        | _ -> 
            let idx = System.Random().Next(0, List.length targetOutputs)

            let fullSingle = 
                forwardFull parameters inputs.[idx] architecture
                |> backPropFull architecture parameters targets.[idx] learningRate loss
                |> weightUpdate parameters

            if iterations % 100 = 99 || iterations = 0
            then 
                let f = forwardFull parameters inputs.[idx] architecture
                printfn "Iteration number: %A\nError: %A\n" (iterations + 1) (getOverallError targets.[idx].[0] f.[1] loss)
            else printf ""

            train fullSingle model maxIterations (iterations + 1)
    
    train initial architecture iterations 0


let runNetworkTest (trainedNetwork : Parameters) (input : float list) (architecture : Layer list) 
    (loss : Loss) (targetOutputs : float list) : float list * float =
    
    let output = 
        forwardFull trainedNetwork input architecture
    
    let error = 
        getOverallError targetOutputs output.[1] loss // Index of 1 since there was 1.0s added to the front
    
    output.[1], error


let runNetwork (trainedNetwork : Parameters) (input : float list) (architecture : Layer list) : float list =
    
    let output =
        forwardFull trainedNetwork input architecture

    output.[1]