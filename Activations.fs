module Activations

type Activation =
    | Relu
    | Sigmoid
    | Tanh
    | Softmax
    | LeakyRelu of float
    | Elu of float

let activateLayer (activation : Activation) (input : float list) : float list =
    match activation with
    | Sigmoid -> 
        List.map (fun x -> 1.0 / (1.0 + exp(-x))) input
    | Relu -> 
        List.map (fun x -> max x 0.0) input
    | Softmax -> 
        let expSum = List.reduce (+) (List.map (fun el -> exp(el)) input)
        List.map (fun x -> exp(x) / expSum) input
    | LeakyRelu(a) -> List.map (fun x -> if x < 0.0 then x * a else x) input
    | Tanh -> List.map tanh input
    | Elu(a) -> List.map (fun x -> if x < 0.0 then a * (exp(x) - 1.0) else x) input

let dActivateLayer (activation : Activation) (input : float list) : float list =
    match activation with
    | Sigmoid -> 
        List.map (fun x -> x * (1.0 - x)) input
    | Relu -> List.map (fun x -> if x > 0.0 then 1.0 else 0.0) input
    | Softmax ->
        let expSum = List.reduce (fun acc el -> acc + exp(el)) input
        List.map (fun el -> (el * (expSum - exp(el))) / (expSum * expSum)) input
    | LeakyRelu(a) -> List.map (fun x -> if x < 0.0 then a else 1.0) input
    | Tanh -> List.map (fun x -> 1.0 - (tanh x) * (tanh x)) input
    | Elu(a) -> List.map (fun x -> if x < 0.0 then a * (exp(x) - 1.0) + a else 1.0) input

