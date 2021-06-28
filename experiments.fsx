//#r "System.Memory"
#I @"C:\Users\cybernetic\.nuget\packages"
#r @"system.memory\4.5.4\lib\netstandard2.0\System.Memory.dll"
#r "System.Runtime.InteropServices"
#r @"C:\Users\cybernetic\source\repos\Prelude\Prelude\bin\Release\net47\prelude.dll"
#r @"D:\Downloads\NeuralNets\onnx\Microsoft.ML.OnnxRuntime.dll"
#r @"naudio\1.7.3\lib\net35\NAudio.dll"
#r @"newtonsoft.json\13.0.1-beta1\lib\netstandard2.0\Newtonsoft.Json.dll"

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
        //|> Array.map (fun ((i, _), (i2, _)) ->  
        //    let left, right = max 0 (i-1), min (volumeMedians.Length-1) (i2+1) 
        //    left, right, right - left + 1) //add 1 second to the left, and one to the right as buffer/extra
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
    if samples.Length/sampleRate < 35 then [|samples|] // number of seconds
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
        | [||] -> 
            printfn "Could not calculate ideal split. performing even split."
            splitInto30secIntervals sampleRate samples 
        | splits -> 
            printfn "Split found: %A" splits
            splitAudioWith splits waveSeconds 
         
  
let loadAudio (audioFile:string) =
    use wave = new Wave.WaveFileReader(audioFile)

    let waveBuffer = wave.ReadArray (int wave.Length) 
    printfn $"Audio duration: {wave.TotalTime}"
    let samples =
        [| for i in 0 .. 2 .. waveBuffer.Length - 2 ->
            float32 (BitConverter.ToInt16(waveBuffer, i))
            / float32 Int16.MaxValue |]

    samples, wave.WaveFormat.SampleRate

//////////////////////////

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

let silenceThreshold = -60f
let splits = calculateSplittingCandidates -60f volumeMedians

//////////////////////////////
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
      
let modelLargeST = @"D:\Downloads\NeuralNets\wav2vec2-large-960h-lv60-self\wav2vec2-large-960h-lv60-self.onnx"
let modelLarge = @"D:\Downloads\NeuralNets\wav2vec2-large-960h\wav2vec2-large-960h.onnx"
let basemodelq = @"D:\Downloads\NeuralNets\wav2vec2-base-960h\wav2vec2-base-960h-quantized.onnx"
let basemodel = @"D:\Downloads\NeuralNets\wav2vec2-base-960h\wav2vec2-base-960h.onnx"

let speechToTextModel = new InferenceSession(modelLargeST)
//let speechToTextModelBase = new InferenceSession(basemodel)

speechToTextModel.OutputMetadata

let transcribe (session:InferenceSession) data =
    let t = Tensors.ArrayTensorExtensions.ToTensor(array2D [|data|])

    use outputs = session.Run [|NamedOnnxValue.CreateFromTensor("input_values", t)|]
    use output = Seq.head outputs
     
    let result = output.AsTensor<float32>()
    let dims = result.Dimensions.ToArray()
     
    decode [| for i in 0..dims.[1] - 1 ->
                [| for j in 0..dims.[2] - 1 -> result.[0, i, j] |] |]
////////////////////////////

let samples, sampleRate = loadAudio audioFile

let splitaudio = splitAudio sampleRate samples    
let tokenized = Array.map standardize splitaudio

splitaudio.Length

let strs = Array.map (transcribe speechToTextModel) tokenized

String.concat " " strs

/////////////////////////////////

let audioToText (waveBuffer:_[]) =
    let samples =
        [| for i in 0 .. 2 .. waveBuffer.Length - 2 ->
            float32 (BitConverter.ToInt16(waveBuffer, i))
            / float32 Int16.MaxValue |]
     
    let splitaudio = splitAudio 16_000 samples    
    let tokenized = Array.map standardize splitaudio
    
    let strs = Array.map (transcribe speechToTextModel) tokenized
    
    String.concat " " strs


#r @"C:\\Users\cybernetic\.nuget\packages\fsharp.control.asyncseq\3.0.3\lib\netstandard2.1\FSharp.Control.AsyncSeq.dll"

open FSharp.Control

open NAudio.Wave
type CurrentText = { ShortSpan : string; LongSpan : string; FullSpan : string }

//type Msg = 
//    | Bytes of byte [] 
//    | Stop 
//    | StopRec
//    | RestartRec
//    | Content of AsyncReplyChannel<CurrentText>

//let proc (msgbox:MailboxProcessor<_>) = 
//    let shortmem = ResizeArray()
//    let longmem = ResizeArray()
//    let fullmem = ResizeArray()

//    let clearMem() =
//        shortmem.Clear()
//        longmem.Clear()
//        fullmem.Clear()

//                   // t = maxlen in seconds
//    let processByteArray (b : ResizeArray<_>) t data currtext =
//        b.AddRange data  
//        if b.Count > 16000 * 2 * t then 
//            let txt = audioToText (b.ToArray())
//            printfn "%s" (currtext + txt)
//            b.Clear()
//            currtext + txt
//        else ""

//    let processBytes b curr = 
//        fullmem.AddRange b
//        //curr
//        { curr with 
//            ShortSpan = processByteArray shortmem 2 b curr.ShortSpan ;
//            LongSpan = processByteArray longmem 10 b curr.LongSpan;} 


//    let rec main curr = async {
//        match! msgbox.Receive() with    
//        | Bytes b -> return! main (processBytes b curr)
//        | Stop -> 
//            clearMem() 
//            return ()
//        | StopRec ->  return! main {curr with FullSpan = audioToText (fullmem.ToArray())}
//        | RestartRec ->
//            clearMem() 
//            return! main {ShortSpan = ""; LongSpan = ""; FullSpan = ""}
//        | Content r -> 
//            r.Reply (curr)
//            return! main curr
//    }

//    main {ShortSpan = ""; LongSpan = ""; FullSpan = ""}

//let sys = MailboxProcessor.Start proc 

//let res = sys.PostAndReply Content 

//let waveSource = new WaveIn(WaveFormat = new WaveFormat(16000, 1)) 
 
//let memstream = new IO.MemoryStream()

//let waveFile = new WaveFileWriter(memstream, waveSource.WaveFormat)
 
//waveSource.DataAvailable.Add (fun ev -> sys.Post(Bytes ev.Buffer))

//waveSource.RecordingStopped.Add(fun _ -> 
//    sys.Post(Stop)
//    waveSource.Dispose()
//    closeAndDispose waveFile
//    closeAndDispose memstream) 

//waveSource.StartRecording()

//sys.Post(StopRec)

//sys.Post(RestartRec)

//waveSource.StopRecording()
   
//res

//////////////////
open NAudio.Wave

let liveTranscribe (memstream:IO.MemoryStream) = 
    memstream.Reset() |> ignore
    let wave = new Wave.WaveFileReader(memstream)

    let waveBuffer = wave.ReadArray (int wave.Length) 

    let samples =
        [| for i in 0 .. 2 .. waveBuffer.Length - 2 ->
            float32 (BitConverter.ToInt16(waveBuffer, i))
            / float32 Int16.MaxValue |]

    printfn $"Sample Rate: {wave.WaveFormat.SampleRate}"
    printfn $"Total Time: {wave.TotalTime}"
 
    let splitaudio = splitAudio wave.WaveFormat.SampleRate samples    
    let tokenized = Array.map standardize splitaudio

    printfn $"Audio Len: {splitaudio.Length}"

    let strs = Array.map (transcribe speechToTextModel) tokenized

    String.concat " " strs


let newrecord () =
    let waveSource = new WaveIn(WaveFormat = new WaveFormat(16000, 1)) 

    let memstream = new IO.MemoryStream()
    let waveFile = new WaveFileWriter(memstream, waveSource.WaveFormat)

    let disposeAll () = 
        printfn "Disposing..."
        waveSource.Dispose()
        closeAndDispose waveFile
        closeAndDispose memstream
        printfn "Done"

    
    let run() =
        waveSource.StopRecording()
    
        let rstr = liveTranscribe memstream
        disposeAll()
        rstr 
    
    waveSource.DataAvailable.Add
        (fun waveEv ->  
            waveFile.Write(waveEv.Buffer, 0, waveEv.BytesRecorded)
            waveFile.Flush())

    waveSource, run
 
let waveSource, run = newrecord()

waveSource.StartRecording()

let rstr = run()

rstr

Diagnostics.Process.Start("https://www.google.com?q=" + Net.WebUtility.UrlEncode(rstr))
 
#r "System.Speech"

let synth = new System.Speech.Synthesis.SpeechSynthesizer()

synth.Rate <- 6

synth.Speak(rstr)

synth.Speak ("bayesian inference")