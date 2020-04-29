module BackPropogation


open Types
open Activations
open Losses


let backPropSingleLayer (targetOutputs : float list) (loss : Loss) (learningRate : float) 
    (backPropPart : float list) (layerIndex : int) (forwardPropParts : float list list) 
    (allWeights : float list list list) (layers : Layer list) : float list list * float list =
    
    let intermediateOutputDeltaSum =
        if layerIndex = 0
        then // Output layer
            forwardPropParts.[layerIndex + 1]
            |> List.map2 (dLossFunction loss targetOutputs.Length) backPropPart // Apply derivative of the loss function
        else // Hidden layers
            List.init forwardPropParts.[layerIndex + 1].Length (fun _ ->
                forwardPropParts.[layerIndex] 
                |> dActivateLayer layers.[layerIndex - 1].activation // Get the individual deltas for each node in the previous layer
                |> List.map2 (*) backPropPart) 
            |> List.mapi (fun index1 deltas ->
                allWeights.[layerIndex]
                |> List.mapi (fun index2 weights -> weights.[index1] * deltas.[index2])) // Take layer weights and multiply by corresponding delta
            |> List.map List.sum // Sum all the values coming into the node
    
    let newWeights =
        forwardPropParts.[layerIndex + 1]
        |> dActivateLayer layers.[layerIndex].activation
        |> List.map2 (*) intermediateOutputDeltaSum // Multiply value by derivative of the activation function (value now equivalent to delta)
        |> List.map (fun delta -> 
            forwardPropParts.[layerIndex + 2] 
            |> List.map (fun out -> learningRate * delta * out)) // learningRate * delta * outPrev
        |> List.map2 (List.map2 (-)) allWeights.[layerIndex + 1] // Subtract value from corresponding weight
    
    newWeights, intermediateOutputDeltaSum


let backPropFull (layers : Layer list) (parameters : Parameters) (targetOutputs : float list list) 
    (learningRate : float) (loss : Loss) (forwardPropParts : float list list) : (float list list * float list) list =  
    
    let weights = 
        List.rev parameters.weights // Reverse so that we are working from the final layer inwards since it is back propagation
    
    List.init (layers.Length) id
    |> List.fold (fun acc index -> // This folds through the network from the outer layer and calculates all the new values for each weight and adds the list containing each layer to an accumulator
        let targetOutputs =
            acc.[0] 
            |> fst 
            |> List.map List.sum
        
        let backPropPart = snd acc.[0]

        let backPropOutput = // Pass the whole network/weights/outputs to avoid passing a bunch of extra parameters to the function because sometimes we need access to information from three layers. Different to forward full since that only requires the output of the previous layer
            backPropSingleLayer targetOutputs loss learningRate backPropPart index forwardPropParts weights layers
        
        [backPropOutput] @ acc) 
            [targetOutputs, (List.concat targetOutputs)]
        
 