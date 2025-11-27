# Separate Speakers with SpeechBrain SepFormer

This repository provides a command-line script, `separate_speakers.py`, that runs SpeechBrain's SepFormer model to perform two-speaker source separation on WAV files.

## Requirements

- Python 3.8+ is required.
- The script checks for these Python packages at startup and offers to install any that are missing:
  - `speechbrain`
  - `torch`
  - `torchaudio`
  - `numpy`
  - `soundfile`
  - `requests`
  - `huggingface_hub`

### Recommended: run inside a virtual environment

Some systems (for example, Debian/Ubuntu with `python3-pip` installed from apt) protect the system Python environment and can raise an "externally-managed-environment" error if you try to install packages globally. To avoid this, create and activate a virtual environment before running the script:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install speechbrain torch torchaudio numpy soundfile requests huggingface_hub
```

When you're done, deactivate the environment with:

```bash
deactivate
```

## Usage

Run the script directly with Python:

```bash
python separate_speakers.py input.wav --output-dir PATH [--model MODEL_NAME] [--sample-rate N]
```

Arguments:

- `input.wav` (required): Path to a mono or stereo WAV file to separate.
- `--output-dir` (optional): Directory where separated WAV files will be saved. Defaults to `separated`.
- `--model` (optional): SpeechBrain separation model identifier to use. Defaults to `speechbrain/sepformer-wsj02mix` but accepts any compatible SpeechBrain separation model.
- `--sample-rate` (optional): Target sample rate for model input. Defaults to `16000` Hz; input audio is resampled if necessary.

### Output files

Separated sources are written as WAV files to the output directory, using the input file's basename:

- `<basename>_speaker1.wav`
- `<basename>_speaker2.wav`

For example, running `python separate_speakers.py overlap_23.wav` produces `overlap_23_speaker1.wav` and `overlap_23_speaker2.wav`.

### Status and errors

The script logs key steps (audio loading, resampling decisions, model loading, separation start, and output paths). It exits with clear error messages if the input file is missing, the model fails to load, or separation encounters an issue.

If torchaudio reports no working audio backend, install torchaudio with soundfile support (or ensure the `soundfile` package is available) so the script can set a usable backend automatically.

## Repository status

This workspace currently has no Git remotes configured. If you need to verify merges with a GitHub repository, add the appropriate remote and fetch updates before comparing branches.
