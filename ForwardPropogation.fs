module ForwardPropogation

open Activations
open Types

let forwardSingleLayer (bias : float) (weights : float list list) (inputs : float list) (activation : Activation) : float list =
    weights
    |> List.map (fun list -> List.map2 (*) list inputs) // Multiply the inputs to a layer with the layer weights (dot product of their vectors)
    |> List.map (fun list -> List.sum list + bias) // Get the net sum and add the bias
    |> activateLayer activation // This used to return both activated and unactivated nodes but I accidentally forgot to use the unactivated nodes during backpropogation, instead dActivating layers as I needed them. I know this is inefficient but it all works and I don't want to change everything

let forwardFull (parameters : Parameters) (inputs : float list) (layers : Layer list) : float list list =
    List.init layers.Length id
    |> List.fold (fun (acc : float list list) index -> // Fold though the *empty* list using the accumulator to store the intermediate activated and unactivated output of all layers
        [forwardSingleLayer parameters.biases.[index] parameters.weights.[index] acc.[0] (layers.[index].activation)]
        @ acc) [inputs]
    |> List.append [List.init (List.last layers).outputDims (fun _ -> 1.0)] // Add 1.0s to the final layer for making back propogation easier
