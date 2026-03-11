# Beacon Biosignals Sleep Staging Challenge 2025

## Task
5-class EEG sleep staging (Wake, N1, N2, N3, REM) from 5-channel polysomnography signals.
Metric: Macro-averaged F1 score.

## Architecture
- **CNN**: ResNet18 (stride hacked on layer4) with adapted 5-channel input
- **Temporal**: BiLSTM (2 layers, 256 hidden, bidirectional) on top of CNN features
- **Attention**: SE Block (Squeeze-and-Excitation) for channel recalibration

## Pipeline
1. Raw EEG → Mel Spectrogram (64x64) via torchaudio
2. Context window of 21 epochs fed to ResNet18
3. BiLSTM captures temporal dependencies across epochs
4. Center epoch prediction via argmax ensemble

## Training
- 5-Fold Cross Validation
- AdamW + OneCycleLR scheduler
- Mixup augmentation (alpha=1.0, 60% probability)
- Label Smoothing (0.1)
- Mixed precision (AMP)
