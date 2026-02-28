import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "whisperx"

package = types.ModuleType("whisperx")
package.__path__ = [str(PACKAGE_DIR)]
sys.modules.setdefault("whisperx", package)

utils_spec = importlib.util.spec_from_file_location("whisperx.utils", PACKAGE_DIR / "utils.py")
utils_module = importlib.util.module_from_spec(utils_spec)
assert utils_spec.loader is not None
utils_spec.loader.exec_module(utils_module)
sys.modules["whisperx.utils"] = utils_module

audio_spec = importlib.util.spec_from_file_location("whisperx.audio", PACKAGE_DIR / "audio.py")
audio_module = importlib.util.module_from_spec(audio_spec)
assert audio_spec.loader is not None
audio_spec.loader.exec_module(audio_module)


def test_load_audio_missing_file_raises_runtime_error(monkeypatch):
    def mock_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=kwargs.get("args") or args[0],
            stderr=b"missing.wav: No such file or directory",
        )

    monkeypatch.setattr(subprocess, "run", mock_run)

    with pytest.raises(RuntimeError, match="Failed to load audio") as exc_info:
        audio_module.load_audio("missing.wav")

    assert "No such file or directory" in str(exc_info.value)


def test_load_audio_empty_decoded_stream_returns_empty_array(monkeypatch):
    captured = {}

    def mock_run(cmd, capture_output, check):
        captured["cmd"] = cmd

        class Result:
            stdout = b""

        return Result()

    monkeypatch.setattr(subprocess, "run", mock_run)

    audio = audio_module.load_audio("empty.wav", sr=22050)

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.size == 0
    assert captured["cmd"][0] == "ffmpeg"
    assert captured["cmd"][captured["cmd"].index("-i") + 1] == "empty.wav"
    assert captured["cmd"][captured["cmd"].index("-ar") + 1] == "22050"
