// Learn more about F# at http://docs.microsoft.com/dotnet/fsharp

open System
open NAudio.Wave
open NAudio
open Prelude.Common
open Prelude.Math
open Newtonsoft.Json
open Microsoft.ML.OnnxRuntime
open Wav2Vec2
 
//forgetting as losssy compression and information. model of the self 
printfn "loading model"
let speechToTextModel = new InferenceSession(modelLargeST)
printfn "done"

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
 
    let splitaudio = Wav2Vec2.splitAudio wave.WaveFormat.SampleRate samples    
    let tokenized = Array.map Utils.standardize splitaudio

    printfn $"Audio Len: {splitaudio.Length}"

    let strs = Array.mapi (transcribe speechToTextModel splitaudio.Length) tokenized

    String.concat " " strs

     
type Recorder () =   
    
    let t = new Timers.Timer(Interval = 1000. * 60. * 1.5)

    let sformat = WaveFormat(16000, 16, 1)
    let mutable waveSource = new WaveInEvent(WaveFormat = sformat)  
    let mutable memstream = new IO.MemoryStream()
    let mutable waveFile = new WaveFileWriter(memstream, waveSource.WaveFormat)
    
    let mutable isSet = true 
    let mutable isRecording = false 
    let mutable str = ""
    
    let disposeRecorders () = 
        printfn "Disposing..."
        closeAndDispose waveFile
        closeAndDispose memstream 
        waveSource.Dispose() 
        printfn "Done"

    let stoprecording() =
        waveSource.StopRecording()
        t.Stop()
        str <- liveTranscribe memstream
        disposeRecorders() 
        isSet <- false
        isRecording <- false

    let disposeAll() =
        disposeRecorders()
        closeAndDispose t 
        

    let attachDataAvailableEvent() =
        waveSource.DataAvailable.Add
            (fun waveEv ->  
                waveFile.Write(waveEv.Buffer, 0, waveEv.BytesRecorded)
                waveFile.Flush())   

    do  attachDataAvailableEvent() 
 
        t.Elapsed.Add(fun _ -> 
            printfn "Stopping record"; 
            stoprecording(); 
            printfn $"{str}")
      
    member __.Reset() =
        memstream <- new IO.MemoryStream()
        waveFile <- new WaveFileWriter(memstream, waveSource.WaveFormat)
        waveSource <- new WaveInEvent(WaveFormat = sformat)
        attachDataAvailableEvent()
    
    member __.Result = str

    member __.StopRecording() = stoprecording()

    member this.StartRecording() =
        if not isSet then 
            this.Reset()
            isSet <- true

        if not isRecording then
            waveSource.StartRecording()
            t.Start()
            isRecording <- true 
            printfn "Recording..."

    member __.Dispose() = disposeAll()

    interface IDisposable with 
        member __.Dispose() = disposeAll()

    
[<EntryPoint>]
let main argv = 
    let mutable instr = ""
    let mutable recorder = new Recorder()
    printfn "Ready..."
    while instr <> "q" do  
        instr <- Console.ReadLine()
        
        match instr with
        | "r" -> recorder.StartRecording() 
        | "s" ->  
           recorder.StopRecording()
           printfn $"Result: {recorder.Result}" 
        | "q" -> 
            speechToTextModel.Dispose()
            recorder.Dispose()
            printfn "Goodbye"
        | _ -> printfn "Invalid command"

    0 // return an integer exit code