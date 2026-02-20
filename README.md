# Engine Malfunction Detector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-FF6F00?logo=tensorflow&logoColor=white)
![GitHub repo size](https://img.shields.io/github/repo-size/jpnacaratti/engine-malfunction-detector)
![GitHub last commit](https://img.shields.io/github/last-commit/jpnacaratti/engine-malfunction-detector)

Audio-based binary classification pipeline to detect potential engine malfunction from sound clips.

This project builds a classifier on top of VGGish embeddings and supports both TensorFlow and TensorFlow Lite inference.

## Overview

The goal is to classify engine audio into:

- `1` -> engine sound with possible fault/noise
- `0` -> healthy/normal engine sound

The pipeline includes:

1. YouTube data discovery (fault-like and healthy engine sounds)
2. Audio download and preprocessing (WAV, 16 kHz)
3. Manual chunk selection and labeling
4. VGGish feature extraction (128-d embeddings)
5. Classifier training and hyperparameter search
6. Validation in TensorFlow and TensorFlow Lite

## Project Pipeline

### 1) Fetch YouTube IDs

Script: `youtube_ids_extractor.py`

- Searches YouTube using predefined queries
- Saves metadata to:
  - `dataset/noise_sounds_ids.csv`
  - `dataset/healthy_sounds_ids.csv`

### 2) Download Audio and Resample

Script: `youtube_sound_extractor.py`

- Downloads audio tracks from YouTube IDs
- Converts to WAV and resamples to 16 kHz
- Outputs to:
  - `dataset/noise_files/`
  - `dataset/healthy_files/`

### 3) Manual Chunk Selection

Script: `audio_selector.py`

- Splits each audio into fixed-size chunks (default: 5 s)
- Plays chunks and allows manual selection (`y/w/n/r`)
- Saves selected chunks to:
  - `dataset/noise_cutted/`
  - `dataset/healthy_cutted/`

### 4) Feature Extraction (VGGish)

Script: `dataset_csv_gen.py`

- Extracts VGGish embeddings from selected chunks
- Builds training table:
  - `dataset/audio_features.csv`
- Output format:
  - `audio_path`
  - `has_noise`
  - `feature_0 ... feature_127`

### 5) Train Classifier

Script: `train_classifier.py`

- Loads `dataset/audio_features.csv`
- Trains a dense neural classifier (`models.py`)
- Saves trained model to `models/`

### 6) Hyperparameter Search (Optional)

Script: `grid_search.py`

- Runs multiple combinations of model/training configs
- Saves model artifacts and metrics into `grid_search/`

### 7) Validate TensorFlow Model

Script: `validating_model.py`

- Runs end-to-end validation with standard TensorFlow model

### 8) Convert to TensorFlow Lite

Script: `tflite_converter.py`

- Converts both classifier and VGGish to `.tflite`

### 9) Validate TensorFlow Lite Pipeline

Script: `validating_tflite.py`

- Validates inference flow using TFLite interpreters

## Repository Structure

```text
engine-malfunction-detector/
├── audio_selector.py
├── dataset_csv_gen.py
├── grid_search.py
├── models.py
├── tflite_converter.py
├── train_classifier.py
├── utils.py
├── validating_model.py
├── validating_tflite.py
├── youtube_ids_extractor.py
├── youtube_sound_extractor.py
├── dataset/
│   ├── healthy_sounds_ids.csv
│   ├── noise_sounds_ids.csv
│   └── audio_features.csv
└── vggish/
```

## Requirements

- Python 3.10 (recommended for TensorFlow 2.10 compatibility)
- FFmpeg installed and available in PATH (used by `yt_dlp`)
- A valid YouTube Data API key

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

1. Add your API key in `youtube_ids_extractor.py`:

```python
DEV_API_KEY = "YOUR_KEY_HERE"
```

2. Run data collection:

```bash
python youtube_ids_extractor.py
python youtube_sound_extractor.py
```

3. Curate chunks manually:

```bash
python audio_selector.py
```

4. Generate embeddings dataset:

```bash
python dataset_csv_gen.py
```

5. Train classifier:

```bash
python train_classifier.py
```

6. (Optional) Run grid search:

```bash
python grid_search.py
```

7. Convert to TFLite:

```bash
python tflite_converter.py
```

8. Validate:

```bash
python validating_model.py
python validating_tflite.py
```

## Notes

- The dataset is built from public YouTube audio snippets and manual chunk curation.
- VGGish model assets/checkpoints are required for embedding extraction and conversion.
- This repository currently uses script-based orchestration (no single CLI pipeline entrypoint yet).

## Known Limitations

- Labels depend on manual curation and may include subjectivity.
- Audio domain shift (different microphones/environments/engines) can reduce generalization.
- YouTube source quality is variable and may add noise unrelated to engine condition.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
