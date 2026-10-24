"""Download Silero VAD, SenseVoice, and Hy-MT2 into models/."""

from __future__ import annotations

import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

from exegesays.paths import (
    models_dir,
    sense_voice_dir,
    translator_dir,
    translator_ready,
    vad_path,
)

VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
)
SENSE_VOICE_URLS = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
)
TRANSLATOR_REPO = "mlx-community/Hy-MT2-7B-4bit"
_USER_AGENT = "exegesays"


def main() -> None:
    models_dir().mkdir(parents=True, exist_ok=True)
    _download_file(VAD_URL, vad_path())
    _download_sense_voice()
    _download_translator()
    print("Models are ready.")


def _download_sense_voice() -> None:
    destination = sense_voice_dir()
    model = destination / "model.int8.onnx"
    tokens = destination / "tokens.txt"
    if model.is_file() and tokens.is_file():
        print(f"exists {destination}")
        return

    archive = models_dir() / "sense-voice.tar.bz2"
    last_error: Exception | None = None
    for url in SENSE_VOICE_URLS:
        try:
            print(f"Downloading {url}")
            _download_file(url, archive, overwrite=True)
            _extract_sense_voice(archive, destination)
            return
        except urllib.error.HTTPError as exc:
            last_error = exc
            print(f"Download failed ({exc.code}) for {url}", file=sys.stderr)
        finally:
            if archive.exists():
                archive.unlink()
    raise SystemExit(f"Could not download SenseVoice: {last_error}")


def _extract_sense_voice(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    wanted = {"model.int8.onnx", "tokens.txt"}
    found: set[str] = set()
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name not in wanted or not member.isfile():
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            target = destination / name
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            found.add(name)
    missing = wanted - found
    if missing:
        raise SystemExit(f"SenseVoice archive is missing {', '.join(sorted(missing))}")
    print(f"Extracted SenseVoice into {destination}")


def _download_translator() -> None:
    destination = translator_dir()
    if translator_ready(destination):
        print(f"exists {destination}")
        return
    print(f"Downloading {TRANSLATOR_REPO}")
    snapshot_download(
        repo_id=TRANSLATOR_REPO,
        local_dir=str(destination),
        ignore_patterns=["imgs/*", "*.png"],
    )


def _download_file(url: str, destination: Path, overwrite: bool = False) -> None:
    if destination.is_file() and destination.stat().st_size > 0 and not overwrite:
        print(f"exists {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length") or 0)
            copied = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                copied += len(chunk)
                if total:
                    print(
                        f"\r{destination.name}: {copied / total:.0%}",
                        end="",
                        flush=True,
                    )
            if total:
                print()
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


if __name__ == "__main__":
    main()
