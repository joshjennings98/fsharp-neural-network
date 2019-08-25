module Types

open Activations

type Layer = {inputDims : int; outputDims : int; activation : Activation}

type Parameters = {weights : float list list list; biases : float list}