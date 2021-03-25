
module Utils

let inline private medianGen (x : 'a []) =
    let sorted = (x |> Array.sort)
    let xlen, xlenh = x.Length, x.Length / 2
    xlen % 2 = 0, sorted.[xlenh], sorted.[xlenh - 1]

let inline medianx two (x : ^a []) =
    if x.Length = 1 then x.[0]
    else
        let iseven, med, medl = medianGen x
        if iseven then (med + medl) /two
        else med

let medianf32 x = medianx 2f x

let varianceAndMeanf32 =
    function
    | x when x = Seq.empty -> 0.0f, 0.0f
    | l ->
        let mean = Seq.average l
        (Seq.sumBy (fun x -> (x - mean) ** 2.f) l) / (float32 (Seq.length l)),
        mean 

let standardize (w: float32 []) =
    let var, mean = varianceAndMeanf32 w
    let stdev = sqrt (var + 1e-5f)
    [| for x in w -> (x - mean) / stdev |]


let argmax vector = Array.indexed vector |> Array.maxBy snd |> fst