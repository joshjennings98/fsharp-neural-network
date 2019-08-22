module NeuralNetwork

type Activation =
    | Relu
    | Sigmoid

type Loss =
    | MSE

type Layer = {inputDims : int; outputDims : int; activation : Activation}

type Parameters = {weights : float list list list; biases : float list}

let activateLayer (activation : Activation) (input : float list) : float list =
    match activation with
    | Sigmoid -> List.map (fun x -> 1.0 / (1.0 + exp(-x))) input
    | Relu -> List.map (fun x -> max x 0.0) input

let dActivateLayer (activation : Activation) (input : float list) : float list =
    match activation with
    | Sigmoid -> List.map (fun x -> x * (1.0 - x)) input
    | Relu -> List.map (fun x -> if x > 0.0 then 1.0 else 0.0) input

let lossFunction (loss : Loss) (n : int) : float -> float -> float =
    match loss with
    | MSE -> fun actual target -> (1.0 / (n |> float)) * (actual - target) ** 2.0

let dLossFunction (loss : Loss) (n : int) : float -> float -> float =
    match loss with
    | MSE -> fun actual target -> (2.0 / (n |> float)) * (actual - target) * -1.0

let getOverallError (targetOutputs : float list) (actualOutputs : float list) (loss : Loss) : float =
    actualOutputs
    |> List.map2 (lossFunction loss targetOutputs.Length) targetOutputs
    |> List.sum

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

let backPropSingleLayer (targetOutputs : float list) (loss : Loss) (learningRate : float) (backPropIntermediateOutputs : float list) (layerIndex : int) (forwardPropIntermediateValues : float list list) (allWeights : float list list list) (layers : Layer list) : float list list * float list =
    let intermediateOutputDeltaSum =
        if layerIndex = 0
        then // Output layer
            forwardPropIntermediateValues.[layerIndex + 1]
            |> List.map2 (dLossFunction loss targetOutputs.Length) backPropIntermediateOutputs // Apply derivative of the loss function
        else // Hidden layers
            List.init forwardPropIntermediateValues.[layerIndex + 1].Length (fun _ ->
                List.map2 (*) backPropIntermediateOutputs (forwardPropIntermediateValues.[layerIndex] |> dActivateLayer layers.[layerIndex - 1].activation)) // Get the individual deltas for each node in the previous (or next?) layer
            |> List.mapi (fun index1 deltas ->
                allWeights.[layerIndex]
                |> List.mapi (fun index2 weights -> weights.[index1] * deltas.[index2])) // Take layer weights and multiply by corresponding delta
            |> List.map List.sum // Sum all the values coming into the node
    let newWeights =
        intermediateOutputDeltaSum
        |> List.map2 (*) (dActivateLayer layers.[layerIndex].activation forwardPropIntermediateValues.[layerIndex + 1]) // Multiply value by derivative of the activation function (value now equivalent to delta)
        |> List.map (fun delta -> List.map (fun out ->
            learningRate * delta * out) forwardPropIntermediateValues.[layerIndex + 2]) // learningRate * delta * outPrev
        |> List.map2 (List.map2 (-)) allWeights.[layerIndex + 1] // Subtract value from corresponding weight
    newWeights, intermediateOutputDeltaSum

let backPropFull (layers : Layer list) (parameters : Parameters) (targetOutputs : float list list) (learningRate : float) (loss : Loss) (forwardPropIntermediates : float list list) : ((float list list * float list) list) =
    let backPropWeights = List.rev parameters.weights // Reverse so that we are working from the final layer inwards since it is back propagation
    List.init (layers.Length) id
    |> List.fold (fun acc index ->
        [backPropSingleLayer (acc.[0] // This folds through the network from the outer layer and calculates all the new values for each weight and adds the list containing each layer to an accumulator
            |> fst // Pass the whole network/weights/outputs to avoid passing a bunch of extra parameters to the function because sometimes we need access to information from three layers. Different to forward full since that only requires the output of the previous layer
            |> List.map List.sum) loss learningRate (acc.[0] |> snd) index forwardPropIntermediates backPropWeights layers] @ acc) [targetOutputs, (targetOutputs |> List.concat)]
    
let weightUpdate (parameters : Parameters) (weights : (float list list * float list) list) : Parameters =
    let newWeights =
        weights
        |> List.map fst // Discard intermediate values leaving only the new weights
    {parameters with weights = newWeights}

let trainNetwork (architecture : Layer list) (targetOutputs : float list) (inputs : float list) (learningRate : float) (loss : Loss) (iterations : int) : Parameters =
    let initial = initialiseNetwork architecture // {weights = [[[0.15; 0.2]; [0.25; 0.3]]; [[0.4; 0.45];[0.5; 0.55]]; [[1.0]; [1.0]]]; biases = [0.35; 0.6]}
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
