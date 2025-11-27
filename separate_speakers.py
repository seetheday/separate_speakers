import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch  # pragma: no cover


DEPENDENCIES = [
    "speechbrain",
    "torch",
    "torchaudio",
    "numpy",
    "soundfile",
    "requests",
    "huggingface_hub",
]
MIN_PYTHON = (3, 8)


def cpu_torchaudio_version(torch_version: str) -> str:
    """Return the CPU-only torchaudio version matching the given torch version."""

    base = torch_version.split("+")[0]
    return f"{base}+cpu"


def check_python_version() -> None:
    if sys.version_info < MIN_PYTHON:
        min_version = ".".join(str(part) for part in MIN_PYTHON)
        raise RuntimeError(
            f"Python {min_version} or newer is required. Detected {sys.version.split()[0]}"
        )


def ensure_dependencies() -> None:
    missing = [pkg for pkg in DEPENDENCIES if importlib.util.find_spec(pkg) is None]
    if not missing:
        return

    print("Missing required packages: " + ", ".join(missing))
    response = input("Install missing packages now? [y/N]: ").strip().lower()
    if response not in {"y", "yes"}:
        raise RuntimeError("Cannot continue without installing required packages.")

    for pkg in missing:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def configure_torchaudio_backend(torchaudio_module) -> None:
    """Ensure torchaudio has a usable backend and log choices."""

    preferred_backend = "soundfile"
    try:
        current = torchaudio_module.get_audio_backend()
    except Exception:  # pragma: no cover  # pylint: disable=broad-except
        current = None

    try:
        available = torchaudio_module.list_audio_backends()
    except Exception:  # pragma: no cover  # pylint: disable=broad-except
        available = []

    if current:
        return

    chosen = None
    if preferred_backend in available:
        chosen = preferred_backend
    elif available:
        chosen = available[0]

    if chosen:
        try:
            torchaudio_module.set_audio_backend(chosen)
            print(f"Using torchaudio backend: {chosen}")
            return
        except Exception:  # pragma: no cover  # pylint: disable=broad-except
            pass

    print(
        "Warning: SpeechBrain could not find any working torchaudio backend. "
        "Install torchaudio with soundfile support or set a backend explicitly."
    )


def patch_hf_hub_download(savedir: Path) -> None:
    """Allow SpeechBrain to call hf_hub_download(use_auth_token=...).

    SpeechBrain attempts to download ``custom.py`` even when the model repo does
    not provide it. Older hub versions also removed the ``use_auth_token``
    parameter. This patch both restores the legacy argument and supplies a
    placeholder ``custom.py`` when Hugging Face returns 404 so SepFormer can
    load.
    """

    try:
        import inspect
        from requests import HTTPError
        import huggingface_hub  # type: ignore
    except ImportError:
        return

    custom_placeholder = savedir / "custom.py"
    if not custom_placeholder.exists():
        savedir.mkdir(parents=True, exist_ok=True)
        custom_placeholder.write_text(
            "# Placeholder custom.py created by separate_speakers.py\n"
        )

    original = huggingface_hub.hf_hub_download
    signature = inspect.signature(original)

    def _patched_hf_hub_download(*args, use_auth_token=None, **kwargs):  # type: ignore[override]
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token

        try:
            return original(*args, **kwargs)
        except HTTPError as exc:  # pragma: no cover
            filename = kwargs.get("filename")
            if filename is None and len(args) >= 2:
                filename = args[1]

            status = getattr(getattr(exc, "response", None), "status_code", None)
            if filename == "custom.py" and status == 404:
                print(
                    "custom.py not found in the model repository; using a local"
                    " placeholder instead."
                )
                return str(custom_placeholder)
            raise

    huggingface_hub.hf_hub_download = _patched_hf_hub_download


def load_libraries():
    global np, sf, torch, F, SepformerSeparation  # noqa: PLW0603

    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore
    import torch  # type: ignore

    try:
        import torchaudio  # type: ignore
    except OSError as exc:  # pragma: no cover
        torch_version = getattr(torch, "__version__", "<torch version>")
        ta_version = cpu_torchaudio_version(torch_version)
        raise RuntimeError(
            "torchaudio failed to load shared libraries. "
            "This often happens when a CUDA build is installed without the "
            "matching GPU libraries. Install the CPU-only torchaudio that "
            "matches your torch version, for example:\n"
            f"  pip install --force-reinstall --no-cache-dir torchaudio=={ta_version} "
            "-f https://download.pytorch.org/whl/torch_stable.html"
        ) from exc

    if not hasattr(torchaudio, "list_audio_backends"):
        # Older torchaudio versions do not expose list_audio_backends. SpeechBrain
        # expects it to exist, so provide a minimal compatibility shim.
        def _list_audio_backends() -> list[str]:  # type: ignore[override]
            backends: list[str] = []
            try:
                current = torchaudio.get_audio_backend()
                if current:
                    backends.append(current)
            except Exception:  # pragma: no cover  # pylint: disable=broad-except
                pass
            return backends

        torchaudio.list_audio_backends = _list_audio_backends  # type: ignore[attr-defined]

    configure_torchaudio_backend(torchaudio)

    patch_hf_hub_download(Path("pretrained_sepformer"))

    import torchaudio.functional as F  # type: ignore
    from speechbrain.inference.separation import (  # type: ignore
        SepformerSeparation,
    )


def load_audio(path: Path, target_sample_rate: int) -> tuple["torch.Tensor", int]:
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


def separate_sources(audio: "torch.Tensor", model_name: str) -> "torch.Tensor":
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

    return validate_separated_output(est_sources)


def save_sources(
    est_sources: "torch.Tensor", output_dir: Path, basename: str, sample_rate: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    num_speakers = est_sources.shape[1]
    for i in range(num_speakers):
        source_audio = est_sources[0, i].detach().cpu().numpy().astype(np.float32)
        output_path = output_dir / f"{basename}_speaker{i + 1}.wav"
        sf.write(str(output_path), source_audio, samplerate=sample_rate)
        print(f"Wrote: {output_path}")


def validate_separated_output(est_sources: "torch.Tensor") -> "torch.Tensor":
    """Ensure the separated output has a valid (batch, speakers, samples) shape.

    SpeechBrain SepFormer should return a 3D tensor shaped
    ``(batch, num_speakers, num_samples)``. If a 2D tensor is returned in the
    shape ``(num_speakers, num_samples)``, add the missing batch dimension. Any
    other shape is treated as an error to avoid writing thousands of files when
    a time dimension is mistaken for the number of speakers.
    """

    if est_sources.ndim == 3 and est_sources.shape[1] <= 10:
        return est_sources

    if est_sources.ndim == 2 and est_sources.shape[0] <= 10:
        print("Detected 2D separation output; assuming shape is (speakers, samples)")
        return est_sources.unsqueeze(0)

    raise RuntimeError(
        "Unexpected separation output shape "
        f"{tuple(est_sources.shape)}; expected (batch, speakers, samples)."
    )


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
        check_python_version()
        ensure_dependencies()
        load_libraries()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(exc)
        sys.exit(1)

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
