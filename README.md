# exegesays

A terminal tool that listens to the microphone, keeps one spoken language, and prints a translation of the other. Speech recognition and translation run on this machine after the models have been downloaded.

By default it keeps Mandarin and prints English. `--to zh` keeps English and prints Chinese.

Recognition finishes when you pause for about 0.6 seconds, when speech switches away from the source language, or when you press <kbd>Enter</kbd> to translate the current buffer immediately.

Each line is translated on its own, and tokens print as they are generated. If the model is unsure, or the line looks like a follow-up, it retries using the last few utterances as context.

## Setup

This project uses Python 3.12.

```bash
uv sync --group download
uv run python scripts/download_models.py
```

The download fetches three local models into `models/` (gitignored):

- Silero VAD
- SenseVoice Small int8, for Mandarin, English, Cantonese, Japanese, and Korean
- Tencent Hy-MT2-7B 4-bit, for Chinese ↔ English translation

The first download needs a network connection. After that, `exegesays` does not use the network.

Hy-MT2 is published under Apache 2.0. Unused older translator folders under `models/` can be deleted.

## Run

```bash
# Default (translate Mandarin Chinese to English)
uv run exegesays

# the reverse direction (translation English to Mandarin Chinese)
uv run exegesays --to zh
```

On macOS, allow microphone access for the terminal in `System Settings > Privacy & Security > Microphone`. The first launch asks for that permission.

```bash
uv run exegesays --list-devices
uv run exegesays --device 2
uv run exegesays --silence 0.8
uv run exegesays --show-source
uv run exegesays --verbose
```

`--to en` (default) keeps Mandarin and prints English. `--to zh` keeps English and prints Chinese.

`--show-source` prints the kept source text above each translation. `--verbose` prints recognition details on stderr. By default, only translations are printed.

Press <kbd>Enter</kbd> at any time to translate whatever is in the current speech buffer, without waiting for a pause.

## Tests

```bash
uv run pytest
```

The tests cover the language filter and do not need the microphone or the model files.
