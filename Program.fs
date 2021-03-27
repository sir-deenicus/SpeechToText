
open DeepSpeechClient
open NAudio.Wave
open System.IO
open System
open System.Diagnostics 

let deepSpeechDir = @"D:\Downloads\NeuralNets\deepspeech"

let sw = Stopwatch() 
     
let transcribeWithDeepSpeech verstr (audioFile:string) =
    use ds = new DeepSpeech(sprintf @"%s\deepspeech-%s-models.pbmm" deepSpeechDir verstr)
    ds.EnableExternalScorer(sprintf @"%s\deepspeech-%s-models.scorer" deepSpeechDir verstr)
    
    use wave = new WaveFileReader(audioFile)
    let waveBuffer = Array.create (int wave.Length) 0uy  
    wave.Read (waveBuffer, 0, waveBuffer.Length) |> ignore

    let samples =
        [| for i in 0 .. 2 .. waveBuffer.Length - 2 -> BitConverter.ToInt16(waveBuffer, i)|]

    let sr = ds.GetModelSampleRate()
    let bw = ds.GetModelBeamWidth() 
    
    printfn "Init done, time taken: %A\nSample Rate: %A | Beam Width: %A | Format: %A | Duration: %A" sw.Elapsed sr bw wave.WaveFormat wave.TotalTime

    ds.SpeechToText(samples, uint32 samples.Length)

[<EntryPoint>]
let main argv =
    if argv.Length = 0 then printfn "No input"; 0
    else 
        printfn "Starting...\nItem: %A" argv
    
        sw.Start() 

        printfn "Please select a version\n[1]: DeepSpeech 0.8.2 | [2]: DeepSpeech 0.9 | [3]: wav2vec2 best | [4]: wav2vec2 large | [5]: wav2vec2 base | [6]: wav2vec2 base quantized"

        let verstr =
            match Console.ReadLine() with
            | "1" -> "0.8.2"
            | "2" -> "0.9.0"
            | "3" -> "wav2vec"
            | "4" -> "wav2vec-large"
            | "5" -> "wav2vec-base"
            | "6" -> "wav2vec-base-quantized"
            | _ -> ""

        if verstr = "" then 0
        else
            let audioFileArg = argv.[0]  

            let audioFile =
                if not (File.Exists audioFileArg) then
                    Path.Combine(deepSpeechDir, audioFileArg)
                else audioFileArg
             
            let txt =
                match verstr with
                | "wav2vec" -> Wav2Vec2.transcribeAudio Wav2Vec2.ModelType.LargeST audioFile
                | "wav2vec-large" -> 
                    Wav2Vec2.transcribeAudio Wav2Vec2.ModelType.Large audioFile
                | "wav2vec-base" -> 
                    Wav2Vec2.transcribeAudio Wav2Vec2.ModelType.Base audioFile
                | "wav2vec-base-quantized" -> 
                    Wav2Vec2.transcribeAudio Wav2Vec2.ModelType.BaseQuantized audioFile
                | _ -> transcribeWithDeepSpeech verstr audioFile 

            printfn "Time Taken: %A\n" sw.Elapsed

            let fn = Path.GetFileNameWithoutExtension audioFile

            IO.File.WriteAllText (IO.Path.Combine(deepSpeechDir, $"{fn}.{verstr}.txt"), txt)

            0 

