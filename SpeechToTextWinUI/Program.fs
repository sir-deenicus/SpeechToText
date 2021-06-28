// Learn more about F# at http://docs.microsoft.com/dotnet/fsharp

open System
open NAudio.Wave

// Define a function to construct a message to print
let from whom =
    sprintf "from %s" whom

[<EntryPoint>]
let main argv =
    let mem = new IO.MemoryStream()
    let sformat = WaveFormat(16000, 16, 1)
    let waveSource = new WaveIn(WaveFormat = sformat)

    waveSource.DataAvailable.Add(fun wIn -> wIn.Buffer ())
    let writer =  new WaveFileWriter(mem, sformat)
     
    waveSource.RecordingStopped.Add(fun _ -> ())
     
    waveSource.StartRecording();

    let message = from "F#" // Call the function
    printfn "Hello world %s" message
    0 // return an integer exit code