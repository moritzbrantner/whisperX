import os
import re
import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
from typing import Dict, Optional, Union
import torch

from .audio import load_audio, SAMPLE_RATE, save_audio

class DiarizationPipeline:
    def __init__(
        self,
        model_name="pyannote/speaker-diarization-3.1",
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.model_name = model_name
        self.model = Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token).to(device)

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
        known_speakers_dir: Optional[str] = None,
        save_speakers_dir: Optional[str] = None,
        profile_similarity_threshold: float = 0.7,
    ):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }

        diarization_output = self.model(
            audio_data, 
            num_speakers = num_speakers, 
            min_speakers=min_speakers, 
            max_speakers=max_speakers, 
            return_embeddings=True)

        if isinstance(diarization_output, tuple):
            segments, embeddings = diarization_output
        else:
            segments = diarization_output
            embeddings = None

        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)        
        diarize_df['duration'] = diarize_df['segment'].apply(lambda x: x.end - x.start)

        if embeddings is not None and len(embeddings) == len(diarize_df):
            diarize_df["embedding"] = [np.asarray(embedding, dtype=np.float32) for embedding in embeddings]
            diarize_df = _apply_known_profiles(
                diarize_df,
                known_speakers_dir=known_speakers_dir,
                save_speakers_dir=save_speakers_dir,
                similarity_threshold=profile_similarity_threshold,
            )
        
        return diarize_df


def _sanitize_speaker_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm <= 0:
        return embedding
    return embedding / norm


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = _normalize_embedding(left)
    right_norm = _normalize_embedding(right)
    return float(np.dot(left_norm, right_norm))


def _load_speaker_profiles(directory: Optional[str]) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
    if not directory or not os.path.isdir(directory):
        return {}

    profiles = {}
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".npz"):
            continue
        path = os.path.join(directory, filename)
        data = np.load(path)
        name = str(data.get("name", os.path.splitext(filename)[0]))
        profiles[name] = {
            "embedding": np.asarray(data["embedding"], dtype=np.float32),
            "count": int(data.get("count", 1)),
        }
    return profiles


def _save_speaker_profiles(directory: str, profiles: Dict[str, Dict[str, Union[np.ndarray, int]]]):
    os.makedirs(directory, exist_ok=True)
    for name, profile in profiles.items():
        path = os.path.join(directory, f"{_sanitize_speaker_name(name)}.npz")
        np.savez(path, name=name, embedding=profile["embedding"], count=profile["count"])


def _compute_local_speaker_profiles(diarize_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    local_profiles = {}
    for speaker, group in diarize_df.groupby("speaker"):
        embeddings = np.stack(group["embedding"].values)
        durations = group["duration"].to_numpy(dtype=np.float32)
        if np.sum(durations) > 0:
            local_profiles[speaker] = np.average(embeddings, axis=0, weights=durations)
        else:
            local_profiles[speaker] = np.mean(embeddings, axis=0)
    return local_profiles


def _next_profile_name(existing_names) -> str:
    max_index = 0
    for name in existing_names:
        match = re.fullmatch(r"PROFILE_(\d+)", name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return f"PROFILE_{max_index + 1:04d}"


def _apply_known_profiles(
    diarize_df: pd.DataFrame,
    known_speakers_dir: Optional[str],
    save_speakers_dir: Optional[str],
    similarity_threshold: float,
) -> pd.DataFrame:
    if "embedding" not in diarize_df.columns or len(diarize_df) == 0:
        return diarize_df

    known_profiles = _load_speaker_profiles(known_speakers_dir)
    save_profiles = _load_speaker_profiles(save_speakers_dir)
    merged_profiles = {**known_profiles, **save_profiles}
    local_profiles = _compute_local_speaker_profiles(diarize_df)

    speaker_map: Dict[str, str] = {}
    used_known_names = set()

    for local_name, local_embedding in local_profiles.items():
        best_name = None
        best_score = similarity_threshold
        for known_name, known_profile in merged_profiles.items():
            if known_name in used_known_names:
                continue
            score = _cosine_similarity(local_embedding, known_profile["embedding"])
            if score >= best_score:
                best_name = known_name
                best_score = score

        if best_name is not None:
            speaker_map[local_name] = best_name
            used_known_names.add(best_name)

    for local_name in local_profiles:
        if local_name in speaker_map:
            continue
        if save_speakers_dir:
            new_name = _next_profile_name(set(merged_profiles.keys()) | set(speaker_map.values()))
            speaker_map[local_name] = new_name
            merged_profiles[new_name] = {"embedding": local_profiles[local_name], "count": 0}
        else:
            speaker_map[local_name] = local_name

    diarize_df["speaker"] = diarize_df["speaker"].replace(speaker_map)

    if save_speakers_dir:
        for local_name, profile_name in speaker_map.items():
            current_embedding = local_profiles[local_name]
            profile = merged_profiles.get(profile_name)
            if profile is None:
                merged_profiles[profile_name] = {"embedding": current_embedding, "count": 1}
                continue

            previous_count = int(profile.get("count", 0))
            previous_embedding = np.asarray(profile["embedding"], dtype=np.float32)
            total = previous_count + 1
            merged_profiles[profile_name] = {
                "embedding": ((previous_embedding * previous_count) + current_embedding) / total,
                "count": total,
            }

        _save_speaker_profiles(save_speakers_dir, merged_profiles)

    return diarize_df


def assign_word_speakers(diarize_df, transcript_result, fill_nearest=False):
    transcript_segments = transcript_result["segments"]
    # create array for speakers
    speakers = {}
    for speaker in diarize_df["speaker"].unique():
        speakers[speaker] = {"duration": 0, "scores": []}
    
    for seg in transcript_segments:
        # assign speaker to segment (if any)
        diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'], seg['start'])
        diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])
        # remove no hit, otherwise we look for closest (even negative intersection...)
        if not fill_nearest:
            dia_tmp = diarize_df[diarize_df['intersection'] > 0]
        else:
            dia_tmp = diarize_df
        if len(dia_tmp) > 0:
            # sum over speakers
            speaker_values = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False)
            seg["speaker"] = speaker_values.index[0] # most common speaker
            seg["speaker_score"] = speaker_values.iloc[0] / speaker_values.sum()     
            
            speakers[speaker_values.index[0]]["duration"] += seg["end"] - seg["start"]
            speakers[speaker_values.index[0]]["scores"].append(seg["speaker_score"]) 
        else:
            seg["speaker"] = "UNIDENTIFIED"
            seg["speaker_score"] = 0
        
        # assign speaker to words
        if 'words' in seg:
            for word in seg['words']:
                if 'start' in word:
                    diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - np.maximum(diarize_df['start'], word['start'])
                    diarize_df['union'] = np.maximum(diarize_df['end'], word['end']) - np.minimum(diarize_df['start'], word['start'])
                    # remove no hit
                    if not fill_nearest:
                        dia_tmp = diarize_df[diarize_df['intersection'] > 0]
                    else:
                        dia_tmp = diarize_df
                    if len(dia_tmp) > 0:
                        # sum over speakers
                        speaker_values = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False)
                        word["speaker"] = speaker_values.index[0] # most common speaker
                        if speaker_values.sum() > 0:
                            word["speaker_score"] = speaker_values.iloc[0] / speaker_values.sum()
                        else:
                            word["speaker_score"] = 0
                    else:
                        word["speaker"] = "UNIDENTIFIED"
                        word["speaker_score"] = 0
    # convert speakers to list
    speakers = [{
        "id": k, 
        "duration": v["duration"], 
        "score": (sum(v["scores"]) / len(v["scores"])) if len(v["scores"]) > 0 else 0
    } for k, v in speakers.items()]
    transcript_result["speakers"] = speakers
    return transcript_result            


class Segment:
    def __init__(self, start, end, speaker="UNIDENTIFIED"):
        self.start = start
        self.end = end
        self.speaker = speaker
