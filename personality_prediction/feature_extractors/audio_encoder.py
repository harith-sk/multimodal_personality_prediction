"""
Audio Encoder - WavLM Feature Extraction (Singleton Pattern)

Extracts audio features using Microsoft's WavLM model.
Model is loaded once and reused across all requests.
"""

import numpy as np
import torch
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import logging

logger = logging.getLogger(__name__)

# Configuration
TARGET_SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AudioEncoder:
    """
    Singleton audio encoder using WavLM.

    Loads WavLM model once and reuses it for all encoding requests.
    """

    def __init__(self):
        """Initialize and load WavLM model."""
        logger.info(f"Loading WavLM model on {DEVICE}...")

        self.device = DEVICE
        self.target_sr = TARGET_SR

        # Load feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base"
        )

        # Load WavLM model
        self.model = WavLMModel.from_pretrained(
            "microsoft/wavlm-base"
        ).to(self.device)

        self.model.eval()

        logger.info("✅ WavLM model loaded successfully")

    @torch.no_grad()
    def encode(self, audio: np.ndarray) -> np.ndarray:
        """
        Encode audio to features using WavLM.

        Args:
            audio: np.ndarray (N,) mono audio @ 16kHz

        Returns:
            np.ndarray: WavLM embedding (768,)
        """
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Feature extraction
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs["input_values"].to(self.device)

        # Forward pass
        outputs = self.model(input_values)

        # Mean pooling over time
        embedding = outputs.last_hidden_state.mean(dim=1)

        # Return as numpy array
        return embedding.squeeze(0).cpu().numpy()


# Singleton instance
_audio_encoder_instance = None


def get_audio_encoder() -> AudioEncoder:
    """
    Get or create the global audio encoder instance (singleton).

    Returns:
        AudioEncoder instance
    """
    global _audio_encoder_instance

    if _audio_encoder_instance is None:
        _audio_encoder_instance = AudioEncoder()
        logger.info("Created AudioEncoder singleton")

    return _audio_encoder_instance


# Legacy function for backward compatibility
def encode_audio(audio: np.ndarray) -> list:
    """
    Legacy function - redirects to singleton instance.

    Args:
        audio: np.ndarray (N,) mono audio @ 16kHz

    Returns:
        list: WavLM embedding (768,) as list
    """
    encoder = get_audio_encoder()
    features = encoder.encode(audio)
    return features.tolist()