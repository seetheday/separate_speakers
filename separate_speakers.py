import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from speechbrain.inference.separation import SepformerSeparation


def load_audio(path: Path, target_sample_rate: int) -> tuple[torch.Tensor, int]:
    print(f"Loading input audio: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    data, sample_rate = sf.read(str(path))

    audio = torch.tensor(data, dtype=torch.float32)

    if audio.ndim == 2:
        if audio.shape[1] == 0:
            raise ValueError("Input audio has zero channels.")
        print("Converting stereo to mono by averaging channels")
        audio = audio.mean(dim=1)

    if audio.ndim != 1:
        raise ValueError("Input audio must be mono or stereo.")

    if torch.max(torch.abs(audio)) > 1.0:
        audio = audio / torch.max(torch.abs(audio))

    if sample_rate != target_sample_rate:
        print(f"Resampling from {sample_rate} Hz to {target_sample_rate} Hz")
        audio = F.resample(audio.unsqueeze(0), orig_freq=sample_rate, new_freq=target_sample_rate)
        audio = audio.squeeze(0)
        sample_rate = target_sample_rate
    else:
        print(f"Sample rate is {sample_rate} Hz; no resampling needed")

    audio = audio.unsqueeze(0)
    return audio, sample_rate


def separate_sources(audio: torch.Tensor, model_name: str) -> torch.Tensor:
    print(f"Loading model: {model_name}")
    try:
        model = SepformerSeparation.from_hparams(source=model_name, savedir="pretrained_sepformer")
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Failed to load model '{model_name}': {exc}") from exc

    print("Starting separation...")
    try:
        with torch.no_grad():
            est_sources = model.separate_batch(audio)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Separation failed: {exc}") from exc

    return est_sources


def save_sources(est_sources: torch.Tensor, output_dir: Path, basename: str, sample_rate: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    num_speakers = est_sources.shape[1]
    for i in range(num_speakers):
        source_audio = est_sources[0, i].detach().cpu().numpy().astype(np.float32)
        output_path = output_dir / f"{basename}_speaker{i + 1}.wav"
        sf.write(str(output_path), source_audio, samplerate=sample_rate)
        print(f"Wrote: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separate two speakers using SpeechBrain SepFormer")
    parser.add_argument("input", type=Path, help="Path to input WAV file")
    parser.add_argument("--output-dir", type=Path, default=Path("separated"), help="Output directory for separated WAVs")
    parser.add_argument("--model", type=str, default="speechbrain/sepformer-wsj02mix", help="SpeechBrain separation model identifier")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate for model input")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        audio, sample_rate = load_audio(args.input, args.sample_rate)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        sys.exit(1)

    try:
        est_sources = separate_sources(audio, args.model)
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

    basename = args.input.stem
    save_sources(est_sources, args.output_dir, basename, sample_rate)

    print("Done.")


if __name__ == "__main__":
    main()
