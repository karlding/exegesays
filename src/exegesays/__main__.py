"""Command-line entry for the microphone interpreter."""

from __future__ import annotations

import argparse
import sys

from exegesays.listen import list_input_devices, run
from exegesays.paths import missing_model_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Listen for one language and print a translation of the other.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print input devices and exit.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input device index. Defaults to the system microphone.",
    )
    parser.add_argument(
        "--silence",
        type=float,
        default=0.6,
        help="Seconds of silence that end an utterance.",
    )
    parser.add_argument(
        "--to",
        choices=["en", "zh"],
        default="en",
        help="Translation output. en (default) keeps Mandarin and prints English. "
        "zh keeps English and prints Chinese.",
    )
    parser.add_argument(
        "--show-source",
        action="store_true",
        help="Print the kept source text above each translation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print recognition and timing details on stderr.",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return
    if args.silence <= 0:
        raise SystemExit("--silence must be greater than 0.")

    missing = missing_model_files()
    if missing:
        print("Missing model files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        print(
            "Download them with: uv run python scripts/download_models.py",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        run(
            device=args.device,
            silence=args.silence,
            show_source=args.show_source,
            verbose=args.verbose,
            target=args.to,
        )
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
