import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "whisperx"


def _load_transcribe_module(monkeypatch):
    package = types.ModuleType("whisperx")
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules["whisperx"] = package

    alignment_module = types.ModuleType("whisperx.alignment")
    alignment_module.align_calls = []

    def align(segments, align_model, align_metadata, audio, device, **kwargs):
        alignment_module.align_calls.append(
            {
                "segments": segments,
                "align_model": align_model,
                "align_metadata": align_metadata,
                "audio_size": int(audio.size),
                "device": device,
                "kwargs": kwargs,
            }
        )
        return {"segments": segments}

    def get_align_model(language, device):
        return object(), {"model_name": f"align-{language}", "device": device}

    def get_align_model_name(language):
        return f"align-{language}"

    alignment_module.align = align
    alignment_module.get_align_model = get_align_model
    alignment_module.get_align_model_name = get_align_model_name
    sys.modules["whisperx.alignment"] = alignment_module

    asr_module = types.ModuleType("whisperx.asr")
    asr_module.load_model_calls = []

    class FakeModel:
        def __init__(self, task, language):
            self.task = task
            self.language = language
            self.transcribe_calls = []

        def transcribe(self, audio, batch_size, chunk_size, print_progress):
            self.transcribe_calls.append(
                {
                    "audio_size": int(audio.size),
                    "batch_size": batch_size,
                    "chunk_size": chunk_size,
                    "print_progress": print_progress,
                }
            )
            if self.task == "translate":
                return {"segments": [{"text": "hello", "start": 0, "end": 1}], "text": "hello", "language": "en"}
            return {"segments": [{"text": "hola", "start": 0, "end": 1}], "text": "hola", "language": self.language or "es"}

    def load_model(model_name, **kwargs):
        asr_module.load_model_calls.append({"model_name": model_name, **kwargs})
        return FakeModel(kwargs.get("task"), kwargs.get("language"))

    asr_module.load_model = load_model
    sys.modules["whisperx.asr"] = asr_module

    audio_module = types.ModuleType("whisperx.audio")

    def load_audio(path):
        return np.ones(8, dtype=np.float32)

    audio_module.load_audio = load_audio
    sys.modules["whisperx.audio"] = audio_module

    diarize_module = types.ModuleType("whisperx.diarize")

    class DiarizationPipeline:
        model_name = "diarize-stub"

        def __init__(self, use_auth_token, device):
            self.use_auth_token = use_auth_token
            self.device = device

        def __call__(self, audio_path, **kwargs):
            return [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "kwargs": kwargs}]

    def assign_word_speakers(diarize_segments, result):
        output = dict(result)
        output["diarize_segments"] = diarize_segments
        return output

    diarize_module.DiarizationPipeline = DiarizationPipeline
    diarize_module.assign_word_speakers = assign_word_speakers
    sys.modules["whisperx.diarize"] = diarize_module

    utils_module = types.ModuleType("whisperx.utils")
    utils_module.LANGUAGES = {"en": "English", "es": "Spanish"}
    utils_module.TO_LANGUAGE_CODE = {"spanish": "es"}
    utils_module.writer_calls = []

    def get_writer(output_format, output_dir):
        def writer(result, audio_path, writer_args):
            utils_module.writer_calls.append(
                {
                    "output_format": output_format,
                    "output_dir": output_dir,
                    "audio_path": audio_path,
                    "writer_args": dict(writer_args),
                    "result": dict(result),
                }
            )

        return writer

    def optional_int(value):
        if value in (None, "None"):
            return None
        return int(value)

    def optional_float(value):
        if value in (None, "None"):
            return None
        return float(value)

    def remove_extension(path):
        return str(Path(path).with_suffix(""))

    def str2bool(value):
        if isinstance(value, bool):
            return value
        lowered = value.lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
        raise ValueError(f"invalid boolean value: {value}")

    utils_module.get_writer = get_writer
    utils_module.optional_float = optional_float
    utils_module.optional_int = optional_int
    utils_module.remove_extension = remove_extension
    utils_module.str2bool = str2bool
    sys.modules["whisperx.utils"] = utils_module

    torch_module = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

    torch_module.cuda = _Cuda()
    torch_module.set_num_threads = lambda n: None
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    spec = importlib.util.spec_from_file_location("whisperx.transcribe", PACKAGE_DIR / "transcribe.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, asr_module, utils_module, alignment_module


def test_cli_generates_transcription_and_translation_outputs_for_all_flags(monkeypatch, tmp_path):
    transcribe_module, asr_module, utils_module, alignment_module = _load_transcribe_module(monkeypatch)

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    known_speakers_dir = tmp_path / "known"
    known_speakers_dir.mkdir()
    save_speakers_dir = tmp_path / "saved"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "whisperx",
            str(audio_path),
            "--model",
            "tiny",
            "--model_dir",
            str(tmp_path / "model-cache"),
            "--device",
            "cpu",
            "--device_index",
            "1",
            "--batch_size",
            "3",
            "--compute_type",
            "float32",
            "--output_dir",
            str(tmp_path / "outputs"),
            "--output_format",
            "json",
            "--verbose",
            "true",
            "--language",
            "Spanish",
            "--interpolate_method",
            "linear",
            "--return_char_alignments",
            "--vad_onset",
            "0.45",
            "--vad_offset",
            "0.31",
            "--chunk_size",
            "25",
            "--diarize",
            "--hf_token",
            "token",
            "--min_speakers",
            "1",
            "--max_speakers",
            "3",
            "--known_speakers_dir",
            str(known_speakers_dir),
            "--save_speakers_dir",
            str(save_speakers_dir),
            "--translate",
            "--recognize_entities",
            "--temperature",
            "0.2",
            "--best_of",
            "4",
            "--beam_size",
            "6",
            "--patience",
            "1.2",
            "--length_penalty",
            "0.7",
            "--suppress_tokens=-1,2",
            "--suppress_numerals",
            "--initial_prompt",
            "start prompt",
            "--condition_on_previous_text",
            "false",
            "--fp16",
            "false",
            "--temperature_increment_on_fallback",
            "0.4",
            "--compression_ratio_threshold",
            "2.0",
            "--logprob_threshold",
            "-0.8",
            "--no_speech_threshold",
            "0.4",
            "--max_line_width",
            "32",
            "--max_line_count",
            "2",
            "--highlight_words",
            "true",
            "--segment_resolution",
            "chunk",
            "--threads",
            "2",
            "--print_progress",
            "true",
        ],
    )

    transcribe_module.cli()

    transcription_json = tmp_path / "sample.json"
    translation_json = tmp_path / "sample.en.json"
    assert transcription_json.exists()
    assert translation_json.exists()

    transcription = json.loads(transcription_json.read_text())
    translation = json.loads(translation_json.read_text())
    assert transcription["model"] == "tiny"
    assert transcription["language"] == "es"
    assert translation["model"] == "tiny"
    assert translation["language"] == "en"

    assert len(asr_module.load_model_calls) == 2
    assert asr_module.load_model_calls[0]["task"] == "transcribe"
    assert asr_module.load_model_calls[1]["task"] == "translate"
    assert asr_module.load_model_calls[0]["asr_options"]["beam_size"] == 6
    assert asr_module.load_model_calls[0]["asr_options"]["suppress_tokens"] == [-1, 2]
    assert asr_module.load_model_calls[0]["vad_options"] == {"vad_onset": 0.45, "vad_offset": 0.31}

    assert len(alignment_module.align_calls) == 1
    assert alignment_module.align_calls[0]["kwargs"]["interpolate_method"] == "linear"
    assert alignment_module.align_calls[0]["kwargs"]["return_char_alignments"] is True

    assert len(utils_module.writer_calls) == 2
    assert utils_module.writer_calls[0]["writer_args"] == {
        "highlight_words": True,
        "max_line_count": 2,
        "max_line_width": 32,
    }
    assert utils_module.writer_calls[1]["writer_args"]["translation"] is True


def test_cli_rejects_word_timing_options_when_no_align(monkeypatch, tmp_path):
    transcribe_module, _, _, _ = _load_transcribe_module(monkeypatch)

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "whisperx",
            str(audio_path),
            "--no_align",
            "--max_line_width",
            "40",
        ],
    )

    with pytest.raises(SystemExit):
        transcribe_module.cli()
