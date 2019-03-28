module NeuralNetwork
open Operators

type learnableParameters = {weights : float list list; biases : float list list}

type data = {inputs : float list; outputs : float list}

type networkLayers = (int * string) list

let sigmoid (x : float) : float = 
  1.0 / (1.0 + System.Math.Exp(-1.0 * x))

let dSigmoid (x : float) : float = 
  x * (1.0 - x)

let relu (x : float) : float =
  match (x > 0.0) with
  | true -> x
  | false -> 0.0

let dRelu (x : float) : float = 
  match (x > 0.0) with
  | true -> 1.0
  | false -> 0.0

let initialiseLayers (neuralArchitecture : networkLayers) (inputs : float list) : learnableParameters = 
  let rnd = System.Random()
  let genRandomList (count : int) (seed : System.Random) : float list =
    List.init count (fun el -> 0.01 * (seed.Next(0, 100) |> float))
  let initialBiases = 
    Array.create (neuralArchitecture.Length) [||]
    |> Array.toList
    |> List.mapi (fun i el -> genRandomList (neuralArchitecture.[i] |> fst) rnd)
    |> List.append [(genRandomList (inputs |> List.length) rnd)]
  let initialWeights =
    let network = 
      [inputs |> List.length] @ (List.map (fun el -> el |> fst) neuralArchitecture) 
    Array.create (neuralArchitecture.Length + 1) [||]
    |> Array.toList
    |> List.mapi (fun i el -> 
      if i-1 = -1 
      then [] 
      else genRandomList ((network.[i-1]) * (network.[i])) rnd)
    |> List.tail
  {weights = initialWeights; biases = initialBiases}

let activateLayer (activation : string) (x : float) : float =
  match activation with
  | "sigmoid" -> sigmoid x
  | "relu" -> relu x
  | _ -> failwithf "Invalid Activation." //ToDo: Replace with results type

let dActivateLayer (activation : string) (x : float) : float =
  match activation with
  | "sigmoid" -> dSigmoid x
  | "relu" -> dRelu x
  | _ -> failwithf "Invalid Activation." //ToDo: Replace with results type

let forwardSingleLayer (biases : float list) (weights : float list) (inputs : float list) (activation : string) : float list * float list =
  let tempOutputs = 
    let mutable count = 0
    Array.create biases.Length 0.0
    |> Array.toList
    |> List.map (fun el ->
      let tempInputs = 
        inputs
        |> List.map (fun el -> 
          count <- count + 1
          el * weights.[count-1])
      List.reduce (+) tempInputs)
    |> List.map2 (fun x y -> x + y) biases      
  tempOutputs, (tempOutputs |> List.map (activateLayer activation))

let forwardFull (network : networkLayers) (parameters : learnableParameters) (inputs : float list) = 
  let mutable tempLayer = []
  Array.create (network |> List.length) []
  |> Array.mapi (fun i el ->
    let prevLayerOutputs = if i = 0 then inputs else tempLayer
    let x = forwardSingleLayer parameters.biases.[i] parameters.weights.[i] prevLayerOutputs (network.[i] |> snd)
    tempLayer <- x |> snd
    x)

let getErrors (targetOutputs : float list) (actualOutputs : float list) : float =
  actualOutputs
  |> List.map2 (fun x y -> 0.5 * (x - y) ** 2.0) targetOutputs
  |> List.reduce (+) 


// Currently sorting out Backpropogation. Completely broken at the moment.

let backPropogation (network : networkLayers) (forwardPropogation : (float list * float list) []) (actualOutput : float list) (inputs : float list) (parameters : learnableParameters) =
  let activated =
    forwardPropogation
    |> Array.rev
  let revNetwork = network |> List.rev

  let weights = parameters.weights |> List.rev

  let mutable tempLayer = []
(*
  printf "%A \n" (0.4 - 0.5 * ((activated.[0] |> snd).[0] - actualOutput.[0]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[0]) * (activated.[1] |> snd).[0])
  printf "%A \n" (0.45 - 0.5 * ((activated.[0] |> snd).[0] - actualOutput.[0]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[0]) * (activated.[1] |> snd).[1])
  printf "%A \n" (0.5 - 0.5 * ((activated.[0] |> snd).[1] - actualOutput.[1]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[1]) * (activated.[1] |> snd).[0])
  printf "%A \n" (0.55 - 0.5 * ((activated.[0] |> snd).[1] - actualOutput.[1]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[1]) * (activated.[1] |> snd).[1])
*)
  let temp1 = (((activated.[0] |> snd).[0] - actualOutput.[0]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[0]) * weights.[0].[0])
  printf "t1: %A \n" temp1
  let temp2 = (((activated.[0] |> snd).[1] - actualOutput.[1]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[0]) * weights.[0].[1])//* weights.[0].[1])
  printf "t2: %A \n" temp2
  //printf "%A \n" (0.15 - 0.5 * ((((activated.[0] |> snd).[0] - actualOutput.[0]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[0]) )))//* (weights.[0].[0])) ))//+ (((activated.[0] |> snd).[0] - actualOutput.[0]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[1]) * (weights.[0].[1]))) )// * (dActivateLayer "sigmoid" (activated.[1] |> snd).[0]) * inputs.[0])
  printf "%A \n" (0.2 - 0.5 * ((activated.[0] |> snd).[0] - actualOutput.[0]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[0]) * (weights.[0].[1]) * (dActivateLayer "sigmoid" (activated.[1] |> snd).[0]) * inputs.[1])
  printf "%A \n" (0.25 - 0.5 * ((activated.[0] |> snd).[1] - actualOutput.[1]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[1]) * (weights.[0].[2]) * (dActivateLayer "sigmoid" (activated.[1] |> snd).[1]) * inputs.[0])
  printf "%A \n" (0.3 - 0.5 * ((activated.[0] |> snd).[1] - actualOutput.[1]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[1]) * (weights.[0].[3]) * (dActivateLayer "sigmoid" (activated.[1] |> snd).[1]) * inputs.[1])

  
  //printf "%A \n" activated
  let test1 =
    activated
    |> Array.mapi (fun i el ->
      tempLayer <- el |> snd
      //printf "%A \n" (List.map (fun el -> (dActivateLayer (revNetwork.[i] |> snd) el) * (el - actualOutput.[i % actualOutput.Length]) * (activated.[i+1] |> snd).[i % actualOutput.Length]) tempLayer)
      //printf "%A \n" tempD
    )
  printf "%A \n" test1
  //"test"

  
  

let learningRate = 0.005
let numEpochs = 10000

let testNetwork = [(2, "sigmoid"); (2, "sigmoid")]
let testInput = [0.05; 0.1]
let testOutputs = [0.01; 0.99]
let initialWeights = [[0.15; 0.2;0.25;0.3];[0.4;0.45;0.5;0.55]]
let initialBiases = [[0.35;0.35];[0.6;0.6]]
let initialParameters = {weights = initialWeights; biases = initialBiases}

let testForwardFullOne = forwardFull testNetwork initialParameters testInput
let testNetworkOutputs = (testForwardFullOne.[testForwardFullOne.Length - 1] |> snd)
let errorsOneLayer = getErrors testOutputs testNetworkOutputs

let backPropagationOne = backPropogation testNetwork testForwardFullOne testOutputs testInput initialParameters

printf "%A \n" (testForwardFullOne)
printf "%A \n" (errorsOneLayer)
//printf "%A \n" (backPropagationOne)
