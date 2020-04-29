module ForwardPropogation


open Activations
open Types


let forwardSingleLayer (bias : float) (weights : float list list) (inputs : float list) (activation : Activation) : float list =
    weights
    |> List.map (fun list -> List.map2 (*) list inputs) // Multiply the inputs to a layer with the layer weights (dot product of their vectors)
    |> List.map (fun list -> List.sum list + bias) // Get the net sum and add the bias
    |> activateLayer activation 


let forwardFull (parameters : Parameters) (inputs : float list) (layers : Layer list) : float list list =
    List.init layers.Length id
    |> List.fold (fun (acc : float list list) index -> // Fold though the *empty* list using the accumulator to store the intermediate activated and unactivated output of all layers
        let biases, weights, activation =
            parameters.biases.[index], parameters.weights.[index], layers.[index].activation
        [forwardSingleLayer biases weights acc.[0] activation] @ acc) 
            [inputs]
    |> List.append [List.init (List.last layers).outputDims float] // Add 1.0s to the final layer for making back propogation easier
