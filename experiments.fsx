#r "System.Memory"
#r "System.Runtime.InteropServices"
#r @"C:\Users\cybernetic\source\repos\Prelude\Prelude\bin\Release\net47\prelude.dll"
#r @"D:\Downloads\NeuralNets\onnx\Microsoft.ML.OnnxRuntime.dll"
#r @"C:\Users\cybernetic\.nuget\packages\naudio\1.7.3\lib\net35\NAudio.dll"
#r @"C:\Users\cybernetic\.nuget\packages\newtonsoft.json\13.0.1-beta1\lib\netstandard2.0\Newtonsoft.Json.dll"
open Newtonsoft.Json
open NAudio
open Microsoft.ML.OnnxRuntime
open System
open Prelude.Common 
open Prelude.Math 

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
    
    let startslice, endslice, endlen = splits.[^0] // get last element
    let appendedLen = endlen + (volumeMedians.Length - endslice)
    if appendedLen <= 35 then  //either adjust last element to include rest of sound or append remainder to end
       splits.[^0] <- startslice, volumeMedians.Length - 1, appendedLen 
       splits
    else 
        Array.append splits
            [| endslice, volumeMedians.Length - 1,
               volumeMedians.Length - endslice |]
      
let rec findMinThresh minthresh volumeMedians = 
    if minthresh < -25f then [||]
    else 
        let splitPoints = calculateSplittingCandidates minthresh volumeMedians
        if Array.exists (fun (_,_,r) -> r > 32) splitPoints then
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
                |> Stats.medianf32 |]

        match findMinThresh -60f volumeMedians with 
        | [||] -> splitInto30secIntervals sampleRate samples 
        | splits -> splitAudioWith splits waveSeconds 
         
  
let loadAudio (audioFile:string) =
    use wave = new Wave.WaveFileReader(audioFile)

    let waveBuffer = wave.ReadArray (int wave.Length) 

    let samples =
        [| for i in 0 .. 2 .. waveBuffer.Length - 2 ->
            float32 (BitConverter.ToInt16(waveBuffer, i))
            / float32 Int16.MaxValue |]

    samples, wave.WaveFormat.SampleRate


let audioFile = IO.Path.Combine(@"D:\Downloads\NeuralNets\deepspeech\", "mathdiff.wav")  

let wave = new Wave.WaveFileReader(audioFile)

let waveBuffer = wave.ReadArray (int wave.Length) 

let sampleRate = wave.WaveFormat.SampleRate

let samples =
    [| for i in 0 .. 2 .. waveBuffer.Length - 2 ->
        float32 (BitConverter.ToInt16(waveBuffer, i))
        / float32 Int16.MaxValue |]
         
let waveSeconds = 
    [|for i in 0..sampleRate..samples.Length - 1 -> 
        samples.[i..i + (sampleRate-1)] |] 
      
Array.concat waveSeconds = samples 
      
let volumeMedians =
    [| for section in waveSeconds ->
        section
        |> Array.map (fun v -> 20f * log10 (abs v))
        |> Stats.medianf32 |]

let silenceThreshold = -90f
let splits = calculateSplittingCandidates -60f volumeMedians


let standardize (w: float32 []) =
    let var, mean = Stats.varianceAndMeanf32 w
    let stdev = sqrt (var + 1e-5f)
    [| for x in w -> (x - mean) / stdev |]


let argmax vector = Array.indexed vector |> Array.maxBy snd |> fst

let vocabJson =
    JsonConvert.DeserializeObject<Dict<string, int>>
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

 
let xlmodel = @"D:\Downloads\NeuralNets\wav2vec2-large-960h-lv60-self\wav2vec2-large-960h-lv60-self-quantized.onnx"
let basemodel = @"D:\Downloads\NeuralNets\wav2vec2-base-960h\wav2vec2-base-960h-quantized.onnx"
let speechToTextModel = new InferenceSession(basemodel)

speechToTextModel.OutputMetadata

let transcribe data =
    let t = Tensors.ArrayTensorExtensions.ToTensor(array2D [|data|])

    use outputs = speechToTextModel.Run [|NamedOnnxValue.CreateFromTensor("input_values", t)|]
    use output = Seq.head outputs
     
    let result = output.AsTensor<float32>()
    let dims = result.Dimensions.ToArray()
     
    decode [| for i in 0..dims.[1] - 1 ->
                [| for j in 0..dims.[2] - 1 -> result.[0, i, j] |] |]


let samples, sampleRate = loadAudio audioFile

let splitaudio = splitAudio sampleRate samples    
let tokenized = Array.map standardize splitaudio

splitaudio.Length
let strs = Array.map transcribe tokenized

String.concat " " strs
