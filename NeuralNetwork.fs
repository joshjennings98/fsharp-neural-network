module NeuralNetwork

open System

type learnableParameters = {weights : float list list; biases : float list}

type networkLayers = (int * string) list

let activateLayer (activation : string) : float list -> float list =
  match activation with
  | "id" -> id
  | "sigmoid" -> List.map (fun x -> 1.0 / (1.0 + Math.Exp(-1.0 * x)))
  | "relu" -> List.map (fun x -> max x 0.0) 
  | _ -> failwithf "Invalid Activation." //ToDo: Replace with results type, add softmax (probably using fold)

let dActivateLayer (activation : string) (x : float) : float =
  match activation with
  | "id" -> 1.0
  | "sigmoid" -> x * (1.0 - x) 
  | "relu" -> if x > 0.0 then x else 0.0
  | _ -> failwithf "Invalid Activation." //ToDo: Replace with results type, add softmax (see above), change to output of float list -> float list

let lossFunction (loss : string) (n : int) : float -> float -> float =
  match loss with
  | "mse" -> fun actual target -> (1.0 / (n |> float)) * (actual - target) ** 2.0
  | _ -> failwithf "Invalid Loss Function." //ToDo: Replace with results type

let dLossFunction (loss : string) (n : int) : float -> float -> float =
  match loss with
  | "mse" -> fun actual target -> (2.0 / (n |> float)) * (actual - target)
  | _ -> failwithf "Invalid Loss Function." //ToDo: Replace with results type

let initialiseLayers (layers : networkLayers) (inputs : float list) : learnableParameters = 
  let genRandomList (size : int) : float list =
    let rnd = System.Random()
    List.init size (fun el -> 0.01 * (rnd.Next(0, 100) |> float))
  let initialBiases = 
    genRandomList layers.Length 
  let initialWeights =
    let network = 
      [inputs |> List.length] @ (List.map (fun el -> el |> fst) layers) 
    List.init (layers.Length + 1) id
    |> List.mapi (fun i el -> 
      if i = 0 then [] else genRandomList (network.[i-1] * network.[i]))
    |> List.tail  
  {weights = initialWeights; biases = initialBiases}

let forwardSingleLayer (biases : float) (weights : float list) (inputs : float list) (activation : string) : float list * float list =
  Array.create (weights.Length / inputs.Length) 0.0
  |> Array.toList
  |> List.mapi (fun i el -> 
    biases + (inputs
    |> List.mapi (fun j el -> 
      el * weights.[i * inputs.Length + j])
    |> List.reduce (+)))
    |> fun el -> 
      el, el |> activateLayer activation

let forwardFull (network : networkLayers) (parameters : learnableParameters) (inputs : float list) = 
  List.init (network |> List.length) id
  |> List.fold (fun acc el ->
    [forwardSingleLayer parameters.biases.[acc.Length - 1] parameters.weights.[acc.Length - 1] (acc.Head |> snd) (network.[acc.Length - 1] |> snd)] @ acc) [inputs, inputs]
  |> List.rev
  |> List.tail

let getOverallError (targetOutputs : float list) (actualOutputs : float list) (loss : string) : float =
  actualOutputs
  |> List.map2 (lossFunction loss targetOutputs.Length) targetOutputs
  |> List.reduce (+) 

let backPropogation (network : networkLayers) (forwardPropogation : (float list * float list) list) (targetOutput : float list) (inputs : float list) (parameters : learnableParameters) (loss : string) (learningRate : float) =
  let activated = forwardPropogation |> List.rev
  let weights = parameters.weights |> List.rev

// (*
  printf "%A \n" (weights.[0].[0] - learningRate * ((dLossFunction loss targetOutput.Length) (activated.[0] |> snd).[0] targetOutput.[0]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[0]) * (activated.[1] |> snd).[0])
  printf "%A \n" (weights.[0].[1] - learningRate * ((dLossFunction loss targetOutput.Length) (activated.[0] |> snd).[0] targetOutput.[0]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[0]) * (activated.[1] |> snd).[1])
  printf "%A \n" (weights.[0].[2] - learningRate * ((dLossFunction loss targetOutput.Length) (activated.[0] |> snd).[1] targetOutput.[1]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[1]) * (activated.[1] |> snd).[0])
  printf "%A \n" (weights.[0].[3] - learningRate * ((dLossFunction loss targetOutput.Length) (activated.[0] |> snd).[1] targetOutput.[1]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[1]) * (activated.[1] |> snd).[1])

  printf "\n"

  printf "%A \n" (weights.[1].[0] - ((learningRate * ((dLossFunction loss targetOutput.Length) (activated.[0] |> snd).[0] targetOutput.[0]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[0]) * weights.[0].[0]) + (learningRate * ((dLossFunction loss targetOutput.Length) (activated.[0] |> snd).[1] targetOutput.[1]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[1]) * weights.[0].[2])) * (dActivateLayer (network.[1] |> snd) (activated.[1] |> snd).[0]) * inputs.[0])
  printf "%A \n" (weights.[1].[1] - ((learningRate * ((dLossFunction loss targetOutput.Length) (activated.[0] |> snd).[0] targetOutput.[0]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[0]) * weights.[0].[0]) + (learningRate * ((dLossFunction loss targetOutput.Length) (activated.[0] |> snd).[1] targetOutput.[1]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[1]) * weights.[0].[2])) * (dActivateLayer (network.[1] |> snd) (activated.[1] |> snd).[0]) * inputs.[1])
  printf "%A \n" (weights.[1].[2] - ((learningRate * ((dLossFunction loss targetOutput.Length) (activated.[0] |> snd).[0] targetOutput.[0]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[0]) * weights.[0].[1]) + (learningRate * ((dLossFunction loss targetOutput.Length) (activated.[0] |> snd).[1] targetOutput.[1]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[1]) * weights.[0].[3])) * (dActivateLayer (network.[1] |> snd) (activated.[1] |> snd).[1]) * inputs.[0])
  printf "%A \n" (weights.[1].[3] - ((learningRate * ((dLossFunction loss targetOutput.Length) (activated.[0] |> snd).[0] targetOutput.[0]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[0]) * weights.[0].[1]) + (learningRate * ((dLossFunction loss targetOutput.Length) (activated.[0] |> snd).[1] targetOutput.[1]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[1]) * weights.[0].[3])) * (dActivateLayer (network.[1] |> snd) (activated.[1] |> snd).[1]) * inputs.[1])
// *)  
  "lol test output"

  
  

let learningRate = 0.5
let numEpochs = 10000
let loss = "mse"

let testNetwork = [(2, "sigmoid"); (2, "sigmoid")]
let testInput = [0.05; 0.1]
let testOutputs = [0.01; 0.99]
let initialWeights = [[0.15; 0.2; 0.25; 0.3];[0.4; 0.45; 0.5; 0.55]]
let initialBiases = [0.35; 0.6]
let initialParameters = {weights = initialWeights; biases = initialBiases}

let testForwardFullOne = forwardFull testNetwork initialParameters testInput
let testNetworkOutputs = (testForwardFullOne.[testForwardFullOne.Length - 1])
let errorsOneLayer = getOverallError testOutputs (testNetworkOutputs |> snd) loss

let backPropagationOne = backPropogation testNetwork testForwardFullOne testOutputs testInput initialParameters loss learningRate

//printf "%A \n" (testForwardFullOne)
//printf "%A \n" (errorsOneLayer)
//printf "%A \n" (backPropagationOne)

(*
printf "%A \n" (0.15 - 0.5 * ((((activated.[0] |> snd).[0] - targetOutput.[0]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[0]) * 0.4) + (((activated.[0] |> snd).[1] - targetOutput.[1]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[1]) * 0.5)) * (dActivateLayer (network.[1] |> snd) (activated.[1] |> snd).[0]) * inputs.[0])
printf "%A \n" (0.2 - 0.5 * ((((activated.[0] |> snd).[0] - targetOutput.[0]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[0]) * 0.4) + (((activated.[0] |> snd).[1] - targetOutput.[1]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[1]) * 0.5)) * (dActivateLayer (network.[1] |> snd) (activated.[1] |> snd).[0]) * inputs.[1])
printf "%A \n" (0.25 - 0.5 * ((((activated.[0] |> snd).[0] - targetOutput.[0]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[0]) * 0.45) + (((activated.[0] |> snd).[1] - targetOutput.[1]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[1]) * 0.55)) * (dActivateLayer (network.[1] |> snd) (activated.[1] |> snd).[1]) * inputs.[0])
printf "%A \n" (0.3 - 0.5 * ((((activated.[0] |> snd).[0] - targetOutput.[0]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[0]) * 0.45) + (((activated.[0] |> snd).[1] - targetOutput.[1]) * (dActivateLayer (network.[0] |> snd) (activated.[0] |> snd).[1]) * 0.55)) * (dActivateLayer (network.[1] |> snd) (activated.[1] |> snd).[1]) * inputs.[1])
*)
