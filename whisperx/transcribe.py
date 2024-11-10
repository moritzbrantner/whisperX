import json
import configargparse as argparse
import gc
import os
import warnings
import json

import numpy as np
import torch

from .alignment import align, get_align_model_name, get_align_model
from .asr import load_model
from .audio import load_audio, save_audio
from .diarize import DiarizationPipeline, assign_word_speakers
from .utils import (LANGUAGES, TO_LANGUAGE_CODE, get_writer, optional_float,
                    optional_int, str2bool, remove_extension)

version = "1.0.0" 

def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        default_config_files=['./whisperx.conf','~/whisperx.conf'])
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="small", help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--device_index", default=0, type=int, help="device index to use for FasterWhisper inference")
    parser.add_argument("--batch_size", default=8, type=int, help="the preferred batch size for inference")
    parser.add_argument("--compute_type", default="float16", type=str, choices=["float16", "float32", "int8"], help="compute type for computation")

    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="all", choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"], help="format of the output file; if not specified, all available formats will be produced")
    
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")

    # alignment params
    parser.add_argument("--interpolate_method", default="nearest", choices=["nearest", "linear", "ignore"], help="For word .srt, method to assign timestamps to non-aligned words, or merge them into neighbouring.")
    parser.add_argument("--no_align", action='store_true', help="Do not perform phoneme alignment")
    parser.add_argument("--return_char_alignments", action='store_true', help="Return character-level alignments in the output json file")

    # vad params
    parser.add_argument("--vad_onset", type=float, default=0.500, help="Onset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected")
    parser.add_argument("--vad_offset", type=float, default=0.363, help="Offset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected.")
    parser.add_argument("--chunk_size", type=int, default=30, help="Chunk size for merging VAD segments. Default is 30, reduce this if the chunk is too long.")

    # diarization params
    parser.add_argument("--diarize", action="store_true", help="Apply diarization to assign speaker labels to each segment/word")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Access Token to access PyAnnote gated models")
    parser.add_argument("--min_speakers", default=None, type=int, help="Minimum number of speakers to in audio file")
    parser.add_argument("--max_speakers", default=None, type=int, help="Maximum number of speakers to in audio file")
    parser.add_argument("--known_speakers_dir", type=str, default=None, help="directory from which to load known speakers for diarization")
    parser.add_argument("--save_speakers_dir", type=str, default=None, help="Directory to save speakers to for diarization")
    
    # translation params
    parser.add_argument("--translate", action="store_true", help="Translate the transcribed text to English")
    
    # NER params
    parser.add_argument("--recognize_entities", action="store_true", help="Recognize named entities in the transcribed text")
    
    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=1.0, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--suppress_numerals", action="store_true", help="whether to suppress numeric symbols and currency symbols during sampling, since wav2vec2 cannot align them correctly")

    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=False, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")

    parser.add_argument("--max_line_width", type=optional_int, default=None, help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=optional_int, default=None, help="(not possible with --no_align) the maximum number of lines in a segment")
    parser.add_argument("--highlight_words", type=str2bool, default=False, help="(not possible with --no_align) underline each word as it is spoken in srt and vtt")
    parser.add_argument("--segment_resolution", type=str, default="sentence", choices=["sentence", "chunk"], help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")

    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")

    parser.add_argument("--print_progress", type=str2bool, default = False, help = "if True, progress will be printed in transcribe() and align() methods.")
    # fmt: on
    
    args = parser.parse_args().__dict__
    
    known_speakers_dir = args.pop("known_speakers_dir")
    save_speakers_dir = args.pop("save_speakers_dir")
    
    model_name: str = args.pop("model")
    batch_size: int = args.pop("batch_size")
    model_dir: str = args.pop("model_dir")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    device: str = args.pop("device")
    device_index: int = args.pop("device_index")
    compute_type: str = args.pop("compute_type")

    # model_flush: bool = args.pop("model_flush")
    os.makedirs(output_dir, exist_ok=True)

    # alignment params
    # align_model: str = args.pop("align_model")
    interpolate_method: str = args.pop("interpolate_method")
    no_align: bool = args.pop("no_align")
    return_char_alignments: bool = args.pop("return_char_alignments")

    hf_token: str = args.pop("hf_token")
    vad_onset: float = args.pop("vad_onset")
    vad_offset: float = args.pop("vad_offset")

    chunk_size: int = args.pop("chunk_size")

    # diarization params
    diarize: bool = args.pop("diarize")
    min_speakers: int = args.pop("min_speakers")
    max_speakers: int = args.pop("max_speakers")

    # translation params    
    translate : str = args.pop("translate")
    
    # NER params
    recognize_entities: bool = args.pop("recognize_entities")

    print_progress: bool = args.pop("print_progress")

    if args["language"] is not None:
        args["language"] = args["language"].lower()
        if args["language"] not in LANGUAGES:
            if args["language"] in TO_LANGUAGE_CODE:
                args["language"] = TO_LANGUAGE_CODE[args["language"]]
            else:
                raise ValueError(f"Unsupported language: {args['language']}")

    if model_name.endswith(".en") and args["language"] != "en":
        if args["language"] is not None:
            warnings.warn(
                f"{model_name} is an English-only model but received '{args['language']}'; using English instead."
            )
        args["language"] = "en"

    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    faster_whisper_threads = 4
    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)
        faster_whisper_threads = threads

    asr_options = {
        "beam_size": args.pop("beam_size"),
        "patience": args.pop("patience"),
        "length_penalty": args.pop("length_penalty"),
        "temperatures": temperature,
        "compression_ratio_threshold": args.pop("compression_ratio_threshold"),
        "log_prob_threshold": args.pop("logprob_threshold"),
        "no_speech_threshold": args.pop("no_speech_threshold"),
        "condition_on_previous_text": False,
        "initial_prompt": args.pop("initial_prompt"),
        "suppress_tokens": [int(x) for x in args.pop("suppress_tokens").split(",")],
        "suppress_numerals": args.pop("suppress_numerals"),
    }

    writer = get_writer(output_format, output_dir)
    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    if no_align:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} not possible with --no_align")
    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    writer_args = {arg: args.pop(arg) for arg in word_options}
    
    
    audio_paths = args.pop("audio")    
    print(">> Processing " + str(len(audio_paths)) + " audio files...")
    
    # Part 1: VAD & ASR Loop
    results = []
    tmp_results = []
    model = load_model(
        model_name, 
        device=device, 
        device_index=device_index, 
        download_root=model_dir, 
        compute_type=compute_type, 
        language=args['language'], 
        asr_options=asr_options, 
        vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset}, 
        task="transcribe", 
        threads=faster_whisper_threads)

    index = 0
    for audio_path in audio_paths:
        index += 1
        # check if json file exists
        result_name = remove_extension(audio_path) + ".json"
        if os.path.exists(result_name):
            # load json file
            with open(result_name, "r") as f:
                json_data = json.load(f)
                # check if json_data contains key "model" and if it is the same as the current model
                
            old_name = json_data["model"] if "model" in json_data else None
            if model_name == old_name:            
                old_language = json_data["language"] if "language" in json_data else None
                if args["language"] is None or args["language"] == old_language:
                    print(f"Skipping {audio_path}, json file already exists.")
                    results.append((json_data, audio_path))
                    audio = None
                    continue

        

        # >> VAD & ASR
        print(">>Performing transcription " + str(index) + "/" + str(len(audio_paths)) + " (" + audio_path + ")")
        audio = load_audio(audio_path)
        for adjusted_batch_size in range(batch_size, 0, -1):
            print(f"Trying batch size {adjusted_batch_size}")
            try:
                result = model.transcribe(audio, batch_size=adjusted_batch_size, chunk_size=chunk_size, print_progress=print_progress)
                result["model"] = model_name
                results.append((result, audio_path))
                if True:
                    with open(result_name, "w") as f:
                        json.dump(result, f)
            except:
                continue
            break
        else:
            print(">> Transcription failed for " + audio_path)
            result = {"segments": [], "text": "", "language": args["language"]}
            result["model"] = model_name
            results.append((result, audio_path))
        
    # Unload Whisper and VAD
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Part 2: Align Loop
    if True:
        print(">> Aligning...")
        tmp_results = results
        index = 0
        results = []
        align_model = None
        last_language = None
        for result, audio_path in tmp_results:
            index += 1
            result_name = remove_extension(audio_path) + ".json"
            # >> Align
            if len(result["segments"]) == 0:
                continue
            
            current_language = result["language"]
            if result.get("align_model", None) == get_align_model_name(current_language):
                print(f"Skipping alignment for {audio_path}, already aligned.")
                results.append((result, audio_path))
                continue

            if current_language != last_language:
                # load new language model
                print(f"New language found ({current_language})! Loading new alignment model...")
                #try:
                align_model, align_metadata = get_align_model(current_language, device)
                last_language = current_language
                #except:
                 #   results.append((result, audio_path))
                  #  continue
            print(">>Performing alignment " + str(index) + "/" + str(len(tmp_results)))
            audio = load_audio(audio_path)
            align_result = align(
                result["segments"], 
                align_model, 
                align_metadata, 
                audio, 
                device, 
                interpolate_method=interpolate_method, 
                return_char_alignments=return_char_alignments, 
                print_progress=print_progress)
            align_result["align_model"] = align_metadata["model_name"]
            result.update(align_result) # merge align_result into result
            if True:
                with open(result_name, "w") as f:
                    json.dump(result, f)        
            results.append((result, audio_path))

        # Unload align model
        if align_model is not None:
            del align_model
            gc.collect()
            torch.cuda.empty_cache()
    

    # Part 3: Diarize
    if diarize:
        if hf_token is None:
            print("Warning, no --hf_token used, needs to be saved in environment variable, otherwise will throw error loading diarization model...")
        tmp_results = results
        index = 0
        results = []
        diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device, )
        for result, audio_path in tmp_results:
            index += 1
            result_name = remove_extension(audio_path) + ".json"
                        
            if result.get("diarize_model", None) == diarize_model.model_name:
                print(f"Skipping diarization for {audio_path}, already diarized.")
                results.append((result, audio_path))
                continue
            
            print(">>Performing diarization " + str(index) + "/" + str(len(tmp_results)))
            diarize_segments = diarize_model(
                audio_path, 
                min_speakers=min_speakers, 
                max_speakers=max_speakers, 
                known_speakers_dir=known_speakers_dir, 
                save_speakers_dir=save_speakers_dir)
            
            diarize_result = assign_word_speakers(diarize_segments, result)
            diarize_result["diarize_model"] = diarize_model.model_name
            results.append((diarize_result, audio_path))
            if True:
                with open(result_name, "w") as f:
                    json.dump(diarize_result, f)
            
        del diarize_model
        gc.collect()
        torch.cuda.empty_cache()
        
    # Part 4: Translations
    if translate: 
        translation_results = []
        translation_model = load_model(
            model_name, 
            device=device, 
            device_index=device_index, 
            download_root=model_dir, 
            compute_type=compute_type, 
            language=args['language'], 
            asr_options=asr_options, 
            vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset}, 
            task="translate", 
            threads=faster_whisper_threads)

        index = 0
        for result, audio_path in results:
            if(result.get("language", "en") == "en"):
                continue
            index += 1
            # check if json file exists
            translation_result_name = remove_extension(audio_path) + ".en.json"
            if os.path.exists(translation_result_name):
            
                # load json file
                with open(translation_result_name, "r") as f:
                    json_data = json.load(f)
                    old_name = json_data["model"] if "model" in json_data else None
                    if model_name == old_name:
                        print(f"Skipping {audio_path}, json file already exists.")
                        translation_results.append((json_data, audio_path))
                        continue
        
            audio = load_audio(audio_path)
            # >> VAD & ASR
            print(">>Performing translation " + str(index) + "/" + str(len(results)) + " (" + audio_path + ")")
            # decreasing loop
            for adjusted_batch_size in range(batch_size, 0, -1):
                try:
                    result = translation_model.transcribe(
                        audio, 
                        batch_size=adjusted_batch_size, 
                        chunk_size=chunk_size, 
                        print_progress=print_progress, 
                        task="translate")
                except:
                    continue
                result["model"] = model_name
                result["language"] = "en"
                translation_results.append((result, audio_path))
                if True:
                    with open(translation_result_name, "w") as f:
                        json.dump(result, f)
                break
            else:
                print(">> Translation failed for " + audio_path)
                result = {"segments": [], "text": "", "language": "en"}

        # Unload Whisper and VAD
        del translation_model
        gc.collect()
        torch.cuda.empty_cache()

    # >> Write
    
    for result, audio_path in results:
        print(">> Writing results for " + audio_path)
        result["version"] = version
        writer(result, audio_path, writer_args)
        
    if translate:
        writer_args.update({"translation": True })        
        for result, audio_path in translation_results:
            print(">> Writing translation for " + audio_path)
            result["version"] = version
            writer(result, audio_path, writer_args)

if __name__ == "__main__":
    cli()
