import os
import re
import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
from typing import Optional, Union
import torch

from .audio import load_audio, SAMPLE_RATE, save_audio

pattern = re.compile(r'.*\.npy')  

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

    def __call__(self, audio: Union[str, np.ndarray], num_speakers=None, min_speakers=None, max_speakers=None, known_speakers_dir=None, save_speakers_dir=None):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }
        if known_speakers_dir is not None:
            # load speaker from match_speakers directory
            known_speakers = np.array([])
            # load every file from directory that ends with .npy
            
            for filename in os.listdir(known_speakers_dir):
                if pattern.match(filename):
                    speaker = filename[: -len(".npy")]
                    array = (np.load(filename))
                    print(speaker)
                    known_speakers.append(array)
                    # other.append((speaker, array.shape[0]))
                    
                
            # concat known_speakers to audio_data
            audio_data['waveform'] = torch.cat([audio_data['waveform']] + known_speakers, dim=0)

        
        segments, embeddings = self.model(
            audio_data, 
            num_speakers = num_speakers, 
            min_speakers=min_speakers, 
            max_speakers=max_speakers, 
            return_embeddings=True)
        # print(segments)
        # print(embeddings.shape)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        if known_speakers_dir is not None:
            # add known speakers to embeddings
            embeddings = np.concatenate([embeddings, np.eye(len(known_speakers))], axis=0)
            embeddings
                            
            # remove known speakers
            diarize_df = diarize_df[diarize_df['speaker'] >= len(audio_data['waveform'])]

        # print the cosine similarity between the embeddings
        print(
            np.dot(embeddings, embeddings.T) / (np.linalg.norm(embeddings, axis=1) 
            * np.linalg.norm(embeddings, axis=1)[:, None])
        )
        
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        
        if save_speakers_dir is None:
            return diarize_df
        
        diarize_df['duration'] = diarize_df['segment'].apply(lambda x: x.end - x.start)
        # print(diarize_df)
        
        # for every speaker, find the longest segment
        speakers = diarize_df.loc[diarize_df.groupby("speaker")["duration"].idxmax()]
        print(speakers)
        # cut out the audio segments for each speaker
        for index, row in speakers.iterrows():
            start, end, speaker = row["start"], row["end"], row["speaker"]
            print(start, end)
            audio_segment = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
            save_audio(f"{speaker}.wav", audio_segment, SAMPLE_RATE)
            # save audio segment to file
            # np.save(f"{save_speakers_dir}/speaker_{speaker}.npy", audio_segment)
        
        return diarize_df


def assign_word_speakers(diarize_df, transcript_result, fill_nearest=False):
    transcript_segments = transcript_result["segments"]
    # create array for speakers
    speakers = {}
    for speaker in diarize_df["speaker"].unique():
        speakers[speaker] = {"duration": 0, "scores": []}
    
    prev_speaker = None
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
