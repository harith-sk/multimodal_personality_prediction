"""
Text Encoder - Whisper + RoBERTa Pipeline (Singleton Pattern)

Combines speech-to-text (Whisper) with text encoding (RoBERTa).
Models are loaded once and reused across all requests.
"""

import whisper
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SpeechTextPipeline:
    """
    Singleton pipeline for speech-to-text-to-features.

    Combines Whisper (transcription) + RoBERTa (text encoding).
    """

    def __init__(self):
        """Initialize and load models."""
        logger.info(f"Loading Whisper + RoBERTa models on {DEVICE}...")

        self.device = DEVICE

        # Load Whisper
        self.whisper_model = whisper.load_model("small", device=self.device)
        logger.info("✅ Whisper model loaded")

        # Load RoBERTa
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.text_model = AutoModel.from_pretrained("roberta-base").to(self.device)
        self.text_model.eval()
        logger.info("✅ RoBERTa model loaded")

    def process(self, wav_path: str) -> dict:
        """
        Process audio: transcribe with Whisper, then encode text with RoBERTa.

        Args:
            wav_path: Path to audio/video file

        Returns:
            Dictionary with:
            - 'transcript': Transcribed text string
            - 'text_features': RoBERTa features (768,) as list
        """
        try:
            # Step 1: Language detection
            audio = whisper.load_audio(wav_path)
            audio = whisper.pad_or_trim(audio)

            mel = whisper.log_mel_spectrogram(audio).to(self.device)
            _, probs = self.whisper_model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)

            if detected_lang != "en":
                logger.warning(f"Non-English audio detected: {detected_lang}")

            # Step 2: Transcription
            result = self.whisper_model.transcribe(
                wav_path,
                language="en",
                task="transcribe",
                fp16=(self.device == "cuda"),
                condition_on_previous_text=False,
                temperature=0.0,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                verbose=None         # None = suppress all output including tqdm bars
            )

            # Collect full transcript
            segments = result.get("segments", [])
            text = " ".join(seg["text"].strip() for seg in segments).strip()

            if not text:
                logger.warning("No speech detected in audio")
                return {
                    "transcript": "",
                    "text_features": np.zeros(768, dtype=np.float32).tolist()
                }

            # Step 3: Text encoding
            text_features = self.encode_text(text)

            return {
                "transcript": text,
                "text_features": text_features.tolist()
            }

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return {
                "transcript": "",
                "text_features": np.zeros(768, dtype=np.float32).tolist()
            }

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using RoBERTa.

        Args:
            text: Text string

        Returns:
            np.ndarray: RoBERTa features (768,)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        outputs = self.text_model(**inputs)

        # Mean pooling
        text_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

        return text_embedding.cpu().numpy()


# Singleton instance
_text_encoder_instance = None


def get_text_encoder() -> SpeechTextPipeline:
    """
    Get or create the global text encoder instance (singleton).

    Returns:
        SpeechTextPipeline instance
    """
    global _text_encoder_instance

    if _text_encoder_instance is None:
        _text_encoder_instance = SpeechTextPipeline()
        logger.info("Created SpeechTextPipeline singleton")

    return _text_encoder_instance


# Legacy function for backward compatibility
def speech_to_text_and_features(wav_path: str) -> dict:
    """
    Legacy function - redirects to singleton instance.

    Args:
        wav_path: Path to audio file

    Returns:
        Dictionary with transcript and text features
    """
    pipeline = get_text_encoder()
    return pipeline.process(wav_path)