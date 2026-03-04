# Multimodal Personality Prediction

A deep learning project that predicts Big Five (OCEAN) personality traits from 15-second videos. This system analyzes three modalities simultaneously—video (visual), audio (paralinguistic), and text (semantic)—using state-of-the-art encoders and a custom Transformer-based cross-modal fusion architecture (TACFN).

**Target Accuracy:** 0.911–0.915 Mean Accuracy on the ChaLearn First Impressions V2 dataset.

---

## 1. The Core Architecture

Because processing raw video during training is prohibitively slow, this project uses a two-stage approach:

1. **Feature Extraction (Data Pipeline):** Raw video, audio, and transcriptions are processed once through heavy pretrained models and saved as compressed mathematical vectors (embeddings) to a local cache.
2. **Training & Fusion:** A lightweight, custom neural network (TACFN) loads these cached vectors and learns to combine them to predict personality scores.

---

## 2. Directory Structure

```text
personality_prediction/
│
├── data_pipeline/           # Converts raw videos into fast-loading tensors
│   ├── build_cache.py       # Runs the 3 encoders on all videos; saves `.pt` files
│   └── diagnostics.py       # Sanity checker: ensures the cache is healthy
│
├── feature_extractors/      # The models that process the raw files
│   ├── audio_encoder.py     # WavLM (Audio → 768-dim vector)
│   ├── text_encoder.py      # Whisper + RoBERTa (Speech-to-Text → 768-dim vector)
│   └── visual_encoder.py    # ResNet50 (16 Video Frames → 2048-dim vector)
│
├── training/                # Where the neural network learns
│   ├── config.py            # Central hyperparameter configuration
│   ├── dataset.py           # PyTorch DataLoader for reading the feature cache
│   ├── losses.py            # Combined MAE + MSE loss function
│   ├── metrics.py           # Calculates the official ChaLearn accuracy metric
│   ├── tacfn_model.py       # Custom Transformer/Cross-Attention architecture
│   ├── train_experiments.py # Runs baseline experiments (E2-E12 & sweeps)
│   ├── train_tacfn.py       # Runs the final Two-Phase training for TACFN (E13)
│   └── plot_metrics.py      # Generates evaluation and comparison graphs
│
└── inference/               # Production-ready code
    └── predict.py           # Takes ONE video, runs extraction + TACFN, prints OCEAN
```

---

## 3. The Encoders

*   **Audio (WavLM):** The direct successor to wav2vec2. It captures tone, pitch, and rhythm (paralinguistics) from 16kHz mono audio.
*   **Text (Whisper + RoBERTa):** Whisper provides highly accurate local transcriptions of the audio. RoBERTa-base encodes the semantic meaning of spoken words, using mean pooling over all tokens.
*   **Visual (ResNet50):** 16 frames are uniformly sampled from the video via OpenCV, passed through ResNet50 with ImageNet weights, and mean-pooled to capture facial expressions and body language.

---

## 4. Training the Network (TACFN)

The heart of the project is the **TACFN** model (Transformer-based Adaptive Cross-modal Fusion Network). Once the dataset is cached, training bypasses the heavy encoders completely.

### The Two-Phase Training Strategy

In `train_tacfn.py`, we use a specific two-phase strategy to stabilize the complex architecture:

*   **Phase 1 (Epoch 1-5 - "The Warm-Up"):** The heavy fusion layers (Transformer and Cross-Attention blocks) are **frozen**. Only the initial linear projection layers (which standardize vector sizes to 256) and the final prediction head are trained. This prevents the large, random initial gradients of the Transformer from hopelessly corrupting the delicate extracted features.
*   **Phase 2 (Epoch 6+ - "End-to-End"):** The complex fusion layers are **unfrozen**, and the entire network trains together at a lower learning rate. The Cross-Attention blocks can now safely learn the complex, subtle relationships between what a person looks like, how they sound, and what they are saying.

---

## 5. Early Stopping & Batch Size

*   **Early Stopping:** Used universally across all experiments. The validation accuracy is monitored (never the test set or training loss). If the model fails to improve by at least 0.01% for a set number of epochs (patience=7 for baselines, 10 for TACFN), training halts to prevent overfitting to the noisy, crowdsourced labels.
*   **Batch Size (32):** Carefully chosen based on the GPU memory footprint of the TACFN model (~200-300MB backwards pass). Because the labels are noisy, smaller batch sizes provide an implicit regularizing effect, which is beneficial for this dataset.

---

## 6. How to Run

*Please refer to the internal Training PC Guide for step-by-step commands on building the cache, running diagnostics, and executing the E2-E13 experiment pipeline.*
