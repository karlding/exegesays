"""Locations of the local model files."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def models_dir() -> Path:
    return project_root() / "models"


def vad_path() -> Path:
    return models_dir() / "silero_vad.onnx"


def sense_voice_dir() -> Path:
    return models_dir() / "sense-voice"


def translator_dir() -> Path:
    return models_dir() / "hy-mt2-7b-4bit"


def translator_ready(path: Path | None = None) -> bool:
    directory = translator_dir() if path is None else path
    if not directory.is_dir() or not (directory / "config.json").is_file():
        return False
    return any(directory.glob("*.safetensors"))


def missing_model_files() -> list[Path]:
    required = [
        vad_path(),
        sense_voice_dir() / "model.int8.onnx",
        sense_voice_dir() / "tokens.txt",
    ]
    missing = [path for path in required if not path.is_file()]
    if not translator_ready():
        missing.append(translator_dir())
    return missing
