module NeuralNetwork
//open Operators

type learnableParameters = {weights : float list list; biases : float list}

type data = {inputs : float list; outputs : float list}

type networkLayers = (int * string) list

let activateLayer (activation : string) (x : float) : float =
  match activation with
  | "id" -> x
  | "sigmoid" -> 1.0 / (1.0 + System.Math.Exp(-1.0 * x)) 
  | "relu" -> max x 0.0 
  | _ -> failwithf "Invalid Activation." //ToDo: Replace with results type

let dActivateLayer (activation : string) (x : float) : float =
  match activation with
  | "id" -> 1.0
  | "sigmoid" -> x * (1.0 - x) 
  | "relu" -> if x > 0.0 then x else 0.0
  | _ -> failwithf "Invalid Activation." //ToDo: Replace with results type

let lossFunction (loss : string) (n : int) : float -> float -> float =
  match loss with
  | "mse" -> fun actual output -> (1.0 / (n |> float)) * (actual - output) ** 2.0
  | _ -> failwithf "Invalid Loss Function." //ToDo: Replace with results type

let initialiseLayers (neuralArchitecture : networkLayers) (inputs : float list) : learnableParameters = 
  let genRandomList (size : int) : float list =
    let rnd = System.Random()
    List.init size (fun el -> 0.01 * (rnd.Next(0, 100) |> float))
  let initialBiases = 
    genRandomList neuralArchitecture.Length 
  let initialWeights =
    let network = 
      [inputs |> List.length] @ (List.map (fun el -> el |> fst) neuralArchitecture) 
    Array.create (neuralArchitecture.Length + 1) [||]
    |> Array.toList
    |> List.mapi (fun i el -> 
      if i = 0 
      then [] 
      else genRandomList (network.[i-1] * network.[i]))
    |> List.tail  
  {weights = initialWeights; biases = initialBiases}

let forwardFull (network : networkLayers) (parameters : learnableParameters) (inputs : float list) = 
  let mutable tempLayer = []
  let forwardSingleLayer (biases : float) (weights : float list) (inputs : float list) (activation : string) : float list * float list =
    //let mutable count = -1
    let tempOutputs = 
      Array.create (weights.Length / inputs.Length) 0.0
      |> Array.toList
      |> List.mapi (fun i el -> 
        biases + (inputs
        |> List.mapi (fun j el -> 
          //count <- count + 1
          el * weights.[i * inputs.Length + j (*count*)])
        |> List.reduce (+)))
    tempOutputs, tempOutputs |> List.map (activateLayer activation)
  Array.create (network |> List.length) []
  |> Array.mapi (fun i el ->
    let prevLayerOutputs = if i = 0 then inputs else tempLayer
    let x = forwardSingleLayer parameters.biases.[i] parameters.weights.[i] prevLayerOutputs (network.[i] |> snd)
    tempLayer <- x |> snd
    x)
  
let getOverallError (targetOutputs : float list) (actualOutputs : float list) (loss : string) : float =
  actualOutputs
  |> List.map2 (lossFunction loss targetOutputs.Length) targetOutputs
  |> List.reduce (+) 

// Currently sorting out Backpropogation. Completely broken at the moment.

let backPropogation (network : networkLayers) (forwardPropogation : (float list * float list) []) (actualOutput : float list) (inputs : float list) (parameters : learnableParameters) =
  let activated = forwardPropogation |> Array.rev
  let revNetwork = network |> List.rev
  let weights = parameters.weights |> List.rev

  let mutable tempLayer = []
// (*
  printf "%A \n" (0.4 - 0.5 * ((activated.[0] |> snd).[0] - actualOutput.[0]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[0]) * (activated.[1] |> snd).[0])
  printf "%A \n" (0.45 - 0.5 * ((activated.[0] |> snd).[0] - actualOutput.[0]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[0]) * (activated.[1] |> snd).[1])
  printf "%A \n" (0.5 - 0.5 * ((activated.[0] |> snd).[1] - actualOutput.[1]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[1]) * (activated.[1] |> snd).[0])
  printf "%A \n" (0.55 - 0.5 * ((activated.[0] |> snd).[1] - actualOutput.[1]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[1]) * (activated.[1] |> snd).[1])
// *)
  let temp1 = (((activated.[0] |> snd).[0] - actualOutput.[0]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[0]) * weights.[0].[0])
  printf "t1: %A \n" temp1
  let temp2 = (((activated.[0] |> snd).[1] - actualOutput.[1]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[0]) * weights.[0].[1])//* weights.[0].[1])
  printf "t2: %A \n" temp2
  //printf "%A \n" (0.15 - 0.5 * ((((activated.[0] |> snd).[0] - actualOutput.[0]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[0]) )))//* (weights.[0].[0])) ))//+ (((activated.[0] |> snd).[0] - actualOutput.[0]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[1]) * (weights.[0].[1]))) )// * (dActivateLayer "sigmoid" (activated.[1] |> snd).[0]) * inputs.[0])
  //printf "%A \n" (0.2 - 0.5 * ((activated.[0] |> snd).[0] - actualOutput.[0]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[0]) * (weights.[0].[1]) * (dActivateLayer "sigmoid" (activated.[1] |> snd).[0]) * inputs.[1])
  //printf "%A \n" (0.25 - 0.5 * ((activated.[0] |> snd).[1] - actualOutput.[1]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[1]) * (weights.[0].[2]) * (dActivateLayer "sigmoid" (activated.[1] |> snd).[1]) * inputs.[0])
  //printf "%A \n" (0.3 - 0.5 * ((activated.[0] |> snd).[1] - actualOutput.[1]) * (dActivateLayer "sigmoid" (activated.[0] |> snd).[1]) * (weights.[0].[3]) * (dActivateLayer "sigmoid" (activated.[1] |> snd).[1]) * inputs.[1])

  
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
let loss = "mse"

let testNetwork = [(2, "sigmoid"); (2, "sigmoid")]
let testInput = [0.05; 0.1]
let testOutputs = [0.01; 0.99]
let initialWeights = [[0.15; 0.2;0.25;0.3];[0.4;0.45;0.5;0.55]]
let initialBiases = [0.35;0.6]
let initialParameters = {weights = initialWeights; biases = initialBiases}

let testForwardFullOne = forwardFull testNetwork initialParameters testInput
let testNetworkOutputs = (testForwardFullOne.[testForwardFullOne.Length - 1] |> snd)
let errorsOneLayer = getOverallError testOutputs testNetworkOutputs loss

let backPropagationOne = backPropogation testNetwork testForwardFullOne testOutputs testInput initialParameters

printf "%A \n" (testForwardFullOne)
printf "%A \n" (errorsOneLayer)
//printf "%A \n" (backPropagationOne)
