"""
feature_extractors/visual_encoder.py
Visual Encoder — ResNet50 Feature Extraction (Singleton Pattern)

Extracts visual features from video frames using ResNet50.
Model is loaded once and reused across all requests.

FIX applied (guide Section 0.3):
  Added ToPILImage() and Resize((224, 224)) to transform.
  Raw video frames from OpenCV are arbitrary resolution numpy arrays.
  ResNet50 requires exactly (224, 224) input — without the resize it crashes.
"""

import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import logging

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VisualEncoder:
    """
    Singleton visual encoder using ResNet50.
    Loads model once and reuses it for all encoding requests.
    """

    def __init__(self):
        logger.info(f"Loading ResNet50 model on {DEVICE}...")
        self.device = DEVICE

        # Load ResNet50 with pretrained ImageNet weights
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = torch.nn.Identity()   # Remove classification head → outputs (2048,)
        self.model = self.model.to(self.device)
        self.model.eval()

        # ── Transform pipeline ────────────────────────────────────────────────
        # ToPILImage : numpy (H, W, C) uint8  → PIL Image   [REQUIRED for arbitrary sizes]
        # Resize     : any resolution          → (224, 224)  [REQUIRED by ResNet50]
        # ToTensor   : PIL Image               → float tensor [0, 1]
        # Normalize  : ImageNet mean/std
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        logger.info("✅ ResNet50 model loaded successfully")

    @torch.no_grad()
    def encode(self, frames: list) -> np.ndarray:
        """
        Encode a list of video frames to a single (2048,) feature vector.

        Args:
            frames: List of np.ndarray frames, each (H, W, 3) RGB uint8.
                    Frames are sampled uniformly from the video by build_cache.py.

        Returns:
            np.ndarray shape (2048,) — mean-pooled ResNet50 features.
        """
        if not frames:
            logger.warning("No frames provided — returning zero vector")
            return np.zeros(2048, dtype=np.float32)

        tensors = []
        for frame in frames:
            if frame is None:
                continue
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            tensors.append(self.transform(frame))

        if not tensors:
            return np.zeros(2048, dtype=np.float32)

        # Stack all frames into one batch and run a single GPU forward pass
        batch = torch.stack(tensors).to(self.device)   # (N, 3, 224, 224)
        with torch.no_grad():
            embeddings = self.model(batch)             # (N, 2048)
        features = embeddings.mean(dim=0)              # (2048,)
        return features.cpu().numpy()                  # (2048,)


# ── Singleton ─────────────────────────────────────────────────────────────────
_visual_encoder_instance = None


def get_visual_encoder() -> VisualEncoder:
    """Get or create the global VisualEncoder singleton."""
    global _visual_encoder_instance
    if _visual_encoder_instance is None:
        _visual_encoder_instance = VisualEncoder()
        logger.info("Created VisualEncoder singleton")
    return _visual_encoder_instance


def encode_visual(frames: list) -> list:
    """Legacy function — redirects to singleton."""
    return get_visual_encoder().encode(frames).tolist()


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    enc = get_visual_encoder()
    # Simulate two random frames of different sizes (as would come from OpenCV)
    fake_frames = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),    # 480p
        np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),  # 1080p
    ]
    out = enc.encode(fake_frames)
    assert out.shape == (2048,), f"Expected (2048,), got {out.shape}"
    print(f"Output shape : {out.shape}")
    print(f"Mean / Std   : {out.mean():.4f} / {out.std():.4f}")
    print("✅ visual_encoder.py OK")