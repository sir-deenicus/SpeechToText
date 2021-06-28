module Wav2Vec2

open NAudio
open NAudio.Wave
open System
open Newtonsoft.Json
open Microsoft.ML.OnnxRuntime
open Utils

type ModelType =
    | Large 
    | Base
    | BaseQuantized
    | LargeST
     
///group indices that are near each other. 0,1,3,5,20 becomes 0,20
let groupTimeIndices (amplitudes : _ []) =
    let rec buildArray prevSecondsIndex index =
        [| let secondsIndex, _ as p = amplitudes.[index]
           if prevSecondsIndex = -1 || secondsIndex - prevSecondsIndex > 6 then
               yield p
           if index + 1 < amplitudes.Length then
               yield! buildArray secondsIndex (index + 1) |]
    if amplitudes.Length = 0 then [||] else buildArray -1 0

let calculateSplittingCandidates silenceThreshold volumeMedians =
    let splits =
        Array.indexed volumeMedians //index s.t. each index stands for position in seconds
        //filter all but sections whose median amplitude < threshold and are near the border of 30 second increments
        |> Array.filter (fun (i, v) -> v <= silenceThreshold && i % 30 <= 6) 
        |> groupTimeIndices 
        |> Array.pairwise
        |> Array.map (fun ((i, _), (i2, _)) -> i, i2, i2 - i + 1) 

    if Array.isEmpty splits then [||]
    else 
        let startslice, endslice, endlen = splits.[^0] // get last element
        let appendedLen = endlen + (volumeMedians.Length - endslice)
        if appendedLen <= 36 then  //either adjust last element to include rest of sound or append remainder to end
           printfn "Adjusting last slice to contain remainder segment"
           splits.[^0] <- startslice, volumeMedians.Length - 1, appendedLen 
           splits
        else 
            printfn "Appending remainder segment"
            Array.append splits
                [| endslice, volumeMedians.Length - 1,
                   volumeMedians.Length - endslice |]
      
let rec findMinThresh minthresh volumeMedians = 
    printfn $"Threshold = {minthresh}"
    if minthresh > -25f then [||]
    else 
        let splitPoints = calculateSplittingCandidates minthresh volumeMedians
        if Array.isEmpty splitPoints 
            || Array.exists (fun (_,_,r) -> r > 36) splitPoints then
            printfn "segment > 36 seconds found. Lowering threshold"
            findMinThresh (minthresh + 5f) volumeMedians
        else splitPoints 

let splitInto30secIntervals sampleRate (samples: _ []) =
    let intervals = samples.Length / (sampleRate * 30)
    Array.splitInto intervals samples

let splitAudioWith splitRanges (waveSeconds:_[][]) =
    [|for (start,stop,_) in splitRanges -> Array.concat waveSeconds.[start..stop]|]


let splitAudio sampleRate (samples:_[]) = 
    if samples.Length/sampleRate < 35 then [|samples|]
    else  
        let waveSeconds = 
            [|for i in 0..sampleRate..samples.Length - 1 -> 
                samples.[i..i + (sampleRate-1)] |] 
                  
        let volumeMedians =
            [| for section in waveSeconds ->
                section
                |> Array.map (fun v -> 20f * log10 (abs v))
                |> medianf32 |]

        match findMinThresh -60f volumeMedians with 
        | [||] -> 
            printfn "Could not calculate ideal split. performing even split."
            splitInto30secIntervals sampleRate samples 
        | splits -> 
            printfn "Split found: %A" splits
            splitAudioWith splits waveSeconds 
         
  
let loadAudio (audioFile:string) =
    use wave = new WaveFileReader(audioFile) 
    let waveBuffer = Array.create (int wave.Length) 0uy  
    wave.Read (waveBuffer, 0, waveBuffer.Length) |> ignore 

    printfn $"Audio duration: {wave.TotalTime}"
    let samples =
        [| for i in 0 .. 2 .. waveBuffer.Length - 2 ->
            float32 (BitConverter.ToInt16(waveBuffer, i))
            / float32 Int16.MaxValue |]
   
    samples, wave.WaveFormat.SampleRate
      

let vocabJson =
    JsonConvert.DeserializeObject<Collections.Generic.Dictionary<string, int>>
        (IO.File.ReadAllText
             @"D:\Downloads\NeuralNets\wav2vec2-large-960h\vocab.json")

let tokens = [|for KeyValue(letter,_) in vocabJson -> letter.ToLower()|]

let decode tensor =
    Array.map (argmax >> fun i -> tokens.[i]) tensor 
    |> Array.fold (fun (prev,str) s -> 
        match s with 
        | _ when s = prev -> s, str
        | "<pad>" -> s, str
        | "|" -> s, str + " "
        | _ -> s, str + s)  ("","")
     |> snd 

 
let modelLargeST = @"D:\Downloads\NeuralNets\wav2vec2-large-960h-lv60-self\wav2vec2-large-960h-lv60-self.onnx"
let modelLarge = @"D:\Downloads\NeuralNets\wav2vec2-large-960h\wav2vec2-large-960h.onnx"
let basemodelq = @"D:\Downloads\NeuralNets\wav2vec2-base-960h\wav2vec2-base-960h-quantized.onnx"
let basemodel = @"D:\Downloads\NeuralNets\wav2vec2-base-960h\wav2vec2-base-960h.onnx"

let transcribe (session:InferenceSession) len i data = 
    printfn $"Segment {i+1} of {len}"
    let t = Tensors.ArrayTensorExtensions.ToTensor(array2D [|data|])

    use outputs = session.Run [|NamedOnnxValue.CreateFromTensor("input_values", t)|]
    use output = Seq.head outputs
     
    let result = output.AsTensor<float32>()
    let dims = result.Dimensions.ToArray()
     
    decode [| for i in 0..dims.[1] - 1 ->
                [| for j in 0..dims.[2] - 1 -> result.[0, i, j] |] |]


let transcribeAudio modelType audioFile =
    let samples, sampleRate = loadAudio audioFile
    let splitaudio = splitAudio sampleRate samples

    let tokenized = Array.map standardize splitaudio

    let transcriber =
        printfn "Loading Weights..."
        match modelType with
        | BaseQuantized -> 
            transcribe (new InferenceSession(basemodelq)) splitaudio.Length
        | Base -> 
            transcribe (new InferenceSession(basemodel)) splitaudio.Length 
        | Large -> 
            transcribe (new InferenceSession(modelLarge)) splitaudio.Length
        | LargeST ->   
            transcribe (new InferenceSession(modelLargeST)) splitaudio.Length
             
    let strs = Array.mapi transcriber tokenized

    String.concat " " strs
