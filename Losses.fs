module Losses

type Loss =
    | MSE
    | CrossEntropy 
    | MAE

let lossFunction (loss : Loss) (n : int) : float -> float -> float =
    match loss with
    | MSE -> 
        fun actual target -> 
            (1.0 / (n |> float)) * (actual - target) ** 2.0
    | CrossEntropy ->
        fun target actual ->
            match target with // If the error is zero and we don't split it up then it might return NaN since we will evaluate log(0)
            | 1.0 -> target * log(actual)
            | 0.0 ->  + (1.0 - target) * log(1.0 - actual)
            | _ ->  (target * log(actual) + (1.0 - target) * log(1.0 - actual))
    | MAE -> 
        fun actual target -> 
            (1.0 / (n |> float)) * (abs (actual - target))

let dLossFunction (loss : Loss) (n : int) : float -> float -> float =
    match loss with
    | MSE -> fun actual target -> (2.0 / (n |> float)) * (actual - target) * -1.0
    | CrossEntropy -> 
        fun target actual -> 
            let actual1 =
                match actual with // Avoid dividing by zero - This method might not be ideal
                | x when -1e-17 < x && x < 1e-17 -> 1e-16
                | x when x > 0.99999999 -> 0.9999995
                | x when x < -0.99999999 -> -0.9999995
                | _ -> actual
            let x = -1.0 * (target / actual1) + ((1.0 - target) / (1.0 - actual1))
            if System.Double.IsNaN (x |> double) // I really don't know why the following error happens. It usually happens after about 47000ish iterations of using catergorical cross entropy.
            then failwithf "This error usually happens if you're running for too many iterations and the error is already zero. Try reducing number of iterations. Some input is nan."
            else x
    | MAE -> 
        fun target actual -> // After too many iterations this loss function also causes some nan errors like catergorical cross entropy
            if actual > target then 1.0 else -1.0

let getOverallError (targetOutputs : float list) (actualOutputs : float list) (loss : Loss) : float =
    actualOutputs
    |> List.map2 (lossFunction loss targetOutputs.Length) targetOutputs
    |> List.sum
