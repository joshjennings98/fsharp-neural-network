module Activations


open Types


let activateLayer (activation : Activation) (input : float list) : float list =
    match activation with
    | Sigmoid -> 
        List.map (fun x -> 1.0 / (1.0 + exp(-x))) input
    | Relu -> 
        List.map (fun x -> max x 0.0) input
    | Softmax -> 
        let expSum = List.reduce (+) (List.map (fun el -> exp(el)) input)
        List.map (fun x -> exp(x) / expSum) input
    | LeakyRelu(a) -> 
        List.map (fun x -> if x < 0.0 then x * a else x) input
    | Tanh -> 
        List.map tanh input
    | Elu(a) -> 
        List.map (fun x -> if x <= 0.0 then a * (exp(x) - 1.0) else x) input
    | Selu(a, b) ->
        List.map (fun x -> if x <= 0.0 then b * a * (exp(x) - 1.0) else b * x) input
    | Softsign ->
        List.map (fun x -> x / (abs x + 1.0)) input
    | Softplus ->
        List.map (fun x -> log (exp(x) + 1.0)) input
    | Exponential ->
        List.map (fun x -> exp(x)) input
    | HardSigmoid ->
        let hardSigmoid (x : float) =
            match x with
            | x when x < -2.5 -> 0.0
            | x when x > 2.5 -> 1.0
            | _ -> 0.2 * x + 0.5
        List.map hardSigmoid input
    | Linear -> input        


let dActivateLayer (activation : Activation) (input : float list) : float list =
    match activation with
    | Sigmoid -> 
        List.map (fun x -> x * (1.0 - x)) input
    | Relu -> 
        List.map (fun x -> if x > 0.0 then 1.0 else 0.0) input
    | Softmax ->
        let expSum = List.reduce (fun acc el -> acc + exp(el)) input
        List.map (fun el -> (el * (expSum - exp(el))) / (expSum * expSum)) input
    | LeakyRelu(a) -> 
        List.map (fun x -> if x < 0.0 then a else 1.0) input
    | Tanh -> 
        List.map (fun x -> 1.0 - (tanh x) * (tanh x)) input
    | Elu(a) -> 
        List.map (fun x -> if x < 0.0 then a * exp(x) else 1.0) input
    | Selu(a, b) ->
        List.map (fun x -> if x <= 0.0 then b * a * exp(x) else b) input
    | Softsign ->
        List.map (fun x -> 1.0 / ((abs x + 1.0) ** 2.0)) input
    | Softplus ->
        List.map (fun x -> exp(x) / (1.0 + exp(x))) input
    | Exponential ->
        List.map (fun x -> exp(x)) input
    | HardSigmoid ->
        let dHardSigmoid (x : float) =
            match x with
            | x when x < -2.5 || x > 2.5 -> 0.0
            | _ -> 0.2
        List.map dHardSigmoid input
    | Linear -> 
        List.map (fun _ -> 1.0) input
