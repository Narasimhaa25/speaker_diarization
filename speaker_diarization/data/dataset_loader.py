"""
data/dataset_loader.py
──────────────────────
Loader for SLUE-VoxCeleb dataset for speaker diarization and identification.

SLUE-VoxCeleb → Combined dataset for both diarization and speaker ID evaluation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import numpy as np
import librosa
from datasets import load_dataset


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class AudioSegment:
    """A labelled audio segment."""
    audio: np.ndarray          # float32, mono
    sample_rate: int
    speaker_id: str
    start_sec: float = 0.0
    end_sec: float = 0.0
    source: str = "slue-voxceleb"


@dataclass
class DiarizationSegment:
    """Diarization annotation segment."""
    meeting_id: str
    start_sec: float
    duration_sec: float
    speaker_id: str

    @property
    def end_sec(self) -> float:
        return self.start_sec + self.duration_sec


# ─── SLUE-VoxCeleb loader ────────────────────────────────────────────────────

class SLUEVoxCelebLoader:
    """
    Loads SLUE-VoxCeleb dataset for speaker diarization and identification.
    
    Dataset: qmeeus/slue-voxceleb
    - Combined VoxCeleb data with diarization annotations
    - Suitable for both embedding training and DER evaluation
    
    Usage:
        loader = SLUEVoxCelebLoader()
        for segment in loader.iter_utterances(split="train"):
            # Process segment
    """

    def __init__(
        self,
        target_sr: int = 16_000,
        max_duration_sec: float = 10.0,
        min_duration_sec: float = 2.0,
        seed: int = 42,
    ):
        self.target_sr = target_sr
        self.max_duration_sec = max_duration_sec
        self.min_duration_sec = min_duration_sec
        self.rng = random.Random(seed)
        self._dataset = None
        self._speaker_index = {}

    def _load_dataset(self, split: str = "train"):
        """Load the SLUE-VoxCeleb dataset from HuggingFace."""
        if self._dataset is None or split not in self._dataset:
            print(f"Loading SLUE-VoxCeleb dataset (split: {split})...")
            try:
                self._dataset = load_dataset("qmeeus/slue-voxceleb", split=split)
                print(f"Loaded {len(self._dataset)} samples from {split} split")
            except Exception as e:
                print(f"Error loading dataset: {e}")
                print("Note: This dataset may require authentication or may not be available.")
                raise
        return self._dataset

    def _process_audio(
        self, 
        audio_dict: dict, 
        speaker_id: str
    ) -> Optional[AudioSegment]:
        """Process audio from dataset format to AudioSegment."""
        try:
            # Extract audio array and sample rate
            audio = np.array(audio_dict['array'], dtype=np.float32)
            sr = audio_dict['sampling_rate']
            
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            
            # Ensure mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)
            
            duration = len(audio) / self.target_sr
            
            # Filter by duration
            if duration < self.min_duration_sec:
                return None
            
            # Trim to max duration
            if duration > self.max_duration_sec:
                max_samples = int(self.max_duration_sec * self.target_sr)
                start = self.rng.randint(0, max(1, len(audio) - max_samples))
                audio = audio[start: start + max_samples]
            
            return AudioSegment(
                audio=audio,
                sample_rate=self.target_sr,
                speaker_id=speaker_id,
                start_sec=0.0,
                end_sec=len(audio) / self.target_sr,
                source="slue-voxceleb",
            )
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None

    def iter_utterances(
        self, 
        split: str = "train",
        shuffle: bool = False,
        max_samples: Optional[int] = None
    ) -> Generator[AudioSegment, None, None]:
        """
        Iterate over utterances in the dataset.
        
        Parameters
        ----------
        split : str
            Dataset split: "train", "validation", or "test"
        shuffle : bool
            Whether to shuffle the dataset
        max_samples : int | None
            Maximum number of samples to yield (None = all)
        
        Yields
        ------
        AudioSegment
        """
        dataset = self._load_dataset(split)
        
        indices = list(range(len(dataset)))
        if shuffle:
            self.rng.shuffle(indices)
        
        if max_samples:
            indices = indices[:max_samples]
        
        count = 0
        for idx in indices:
            sample = dataset[idx]
            
            # Extract speaker ID (adjust based on actual dataset structure)
            speaker_id = sample.get('speaker_id', sample.get('label', f'speaker_{idx}'))
            if isinstance(speaker_id, int):
                speaker_id = f'speaker_{speaker_id:04d}'
            
            segment = self._process_audio(sample['audio'], str(speaker_id))
            if segment is not None:
                yield segment
                count += 1
        
        print(f"Yielded {count} valid segments from {split} split")

    def get_speaker_ids(self, split: str = "train") -> List[str]:
        """Get list of unique speaker IDs in the dataset."""
        dataset = self._load_dataset(split)
        speaker_ids = set()
        
        for sample in dataset:
            speaker_id = sample.get('speaker_id', sample.get('label', 'unknown'))
            if isinstance(speaker_id, int):
                speaker_id = f'speaker_{speaker_id:04d}'
            speaker_ids.add(str(speaker_id))
        
        return sorted(list(speaker_ids))

    def sample_by_speaker(
        self,
        n_speakers: int,
        utterances_per_speaker: int,
        split: str = "train"
    ) -> List[AudioSegment]:
        """
        Sample a balanced subset of utterances.
        
        Useful for threshold tuning and evaluation.
        
        Parameters
        ----------
        n_speakers : int
            Number of speakers to sample
        utterances_per_speaker : int
            Number of utterances per speaker
        split : str
            Dataset split to sample from
        
        Returns
        -------
        List[AudioSegment]
        """
        dataset = self._load_dataset(split)
        
        # Group by speaker
        by_speaker = {}
        for idx, sample in enumerate(dataset):
            speaker_id = sample.get('speaker_id', sample.get('label', f'speaker_{idx}'))
            if isinstance(speaker_id, int):
                speaker_id = f'speaker_{speaker_id:04d}'
            speaker_id = str(speaker_id)
            
            if speaker_id not in by_speaker:
                by_speaker[speaker_id] = []
            by_speaker[speaker_id].append(sample)
        
        # Sample speakers
        available_speakers = list(by_speaker.keys())
        if len(available_speakers) < n_speakers:
            print(f"Warning: Only {len(available_speakers)} speakers available, "
                  f"requested {n_speakers}")
            n_speakers = len(available_speakers)
        
        selected_speakers = self.rng.sample(available_speakers, n_speakers)
        
        # Sample utterances per speaker
        segments = []
        for speaker_id in selected_speakers:
            utterances = by_speaker[speaker_id]
            n_samples = min(utterances_per_speaker, len(utterances))
            sampled = self.rng.sample(utterances, n_samples)
            
            for sample in sampled:
                segment = self._process_audio(sample['audio'], speaker_id)
                if segment is not None:
                    segments.append(segment)
        
        print(f"Sampled {len(segments)} segments from {n_speakers} speakers")
        return segments


# ─── Backward compatibility aliases ──────────────────────────────────────────

# For backward compatibility with existing code
VoxCelebLoader = SLUEVoxCelebLoader
RTTMEntry = DiarizationSegment
