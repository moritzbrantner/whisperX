import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "whisperx"

package = types.ModuleType("whisperx")
package.__path__ = [str(PACKAGE_DIR)]
sys.modules.setdefault("whisperx", package)

# Stub pyannote dependency so diarize module can be imported without full runtime deps.
pyannote_module = types.ModuleType("pyannote")
pyannote_audio_module = types.ModuleType("pyannote.audio")


class _PipelineStub:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        raise RuntimeError("Pipeline should not be instantiated in this test")


pyannote_audio_module.Pipeline = _PipelineStub
sys.modules["pyannote"] = pyannote_module
sys.modules["pyannote.audio"] = pyannote_audio_module

utils_spec = importlib.util.spec_from_file_location("whisperx.utils", PACKAGE_DIR / "utils.py")
utils_module = importlib.util.module_from_spec(utils_spec)
assert utils_spec.loader is not None
utils_spec.loader.exec_module(utils_module)
sys.modules["whisperx.utils"] = utils_module

audio_spec = importlib.util.spec_from_file_location("whisperx.audio", PACKAGE_DIR / "audio.py")
audio_module = importlib.util.module_from_spec(audio_spec)
assert audio_spec.loader is not None
audio_spec.loader.exec_module(audio_module)
sys.modules["whisperx.audio"] = audio_module

diag_spec = importlib.util.spec_from_file_location("whisperx.diarize", PACKAGE_DIR / "diarize.py")
diag_module = importlib.util.module_from_spec(diag_spec)
assert diag_spec.loader is not None
diag_spec.loader.exec_module(diag_module)


def test_apply_known_profiles_reuses_matching_speaker_name(tmp_path):
    known_dir = tmp_path / "known"
    known_dir.mkdir()
    np.savez(known_dir / "Alice.npz", name="Alice", embedding=np.array([1.0, 0.0], dtype=np.float32), count=3)

    diarize_df = pd.DataFrame(
        {
            "speaker": ["SPEAKER_00", "SPEAKER_00"],
            "duration": [1.0, 2.0],
            "embedding": [
                np.array([0.99, 0.01], dtype=np.float32),
                np.array([0.98, 0.02], dtype=np.float32),
            ],
        }
    )

    result = diag_module._apply_known_profiles(
        diarize_df=diarize_df,
        known_speakers_dir=str(known_dir),
        save_speakers_dir=None,
        similarity_threshold=0.7,
    )

    assert set(result["speaker"]) == {"Alice"}


def test_apply_known_profiles_creates_and_updates_saved_profiles(tmp_path):
    save_dir = tmp_path / "profiles"

    diarize_df = pd.DataFrame(
        {
            "speaker": ["SPEAKER_00", "SPEAKER_01"],
            "duration": [1.0, 1.0],
            "embedding": [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            ],
        }
    )

    result = diag_module._apply_known_profiles(
        diarize_df=diarize_df,
        known_speakers_dir=None,
        save_speakers_dir=str(save_dir),
        similarity_threshold=0.95,
    )

    assert set(result["speaker"]) == {"PROFILE_0001", "PROFILE_0002"}
    saved_files = sorted(p.name for p in save_dir.glob("*.npz"))
    assert saved_files == ["PROFILE_0001.npz", "PROFILE_0002.npz"]

    second_df = pd.DataFrame(
        {
            "speaker": ["SPEAKER_X"],
            "duration": [1.0],
            "embedding": [np.array([1.0, 0.0], dtype=np.float32)],
        }
    )

    second = diag_module._apply_known_profiles(
        diarize_df=second_df,
        known_speakers_dir=None,
        save_speakers_dir=str(save_dir),
        similarity_threshold=0.95,
    )

    assert set(second["speaker"]) == {"PROFILE_0001"}
