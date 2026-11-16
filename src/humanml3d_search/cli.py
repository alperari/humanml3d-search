"""Command-line stages for reproducible HumanML3D activity retrieval."""

import argparse
import json
from pathlib import Path
import sys

from .common import Config


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--version", action="version", version="humanml3d-search 0.1.0")
    commands = result.add_subparsers(dest="command", required=True)
    for name, help_text in (("index", "Index original captions, split membership and motion availability"),
                            ("model", "Download/check the exact MPNet model snapshot"),
                            ("search", "Run lexical search, semantic top-k and their motion union"),
                            ("visualize", "Build a local video gallery for a completed search"),
                            ("run", "Index, search and create a review gallery")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, default=Path("configs/local.yml"))
        if name in ("index", "run"):
            command.add_argument("--refresh", action="store_true", help="Re-read source files instead of reusing the index snapshot")
        if name in ("model", "search", "run"):
            command.add_argument("--offline", action="store_true", help="Require the pinned model to exist in the local cache")
        if name in ("search", "run"):
            command.add_argument("--activity", default="push_up", help="Activity key in activities.yml")
            command.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
        if name == "visualize":
            command.add_argument("--run-dir", type=Path, required=True, help="Directory containing run_manifest.json")
        if name in ("visualize", "run"):
            command.add_argument("--copy-videos", action="store_true", help="Copy preview MP4s for a portable gallery; default uses symlinks")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = Config.load(args.config)
        if args.command in ("index", "run"):
            from .indexing import build_index

            report = build_index(config, refresh=args.refresh)
            print(json.dumps({"index_manifest": str(config.cache_root / "index.json"), "counts": report["counts"]}, indent=2), flush=True)
        if args.command == "model":
            from .embeddings import fetch_model

            report = fetch_model(config, offline=args.offline)
            print(json.dumps({"model": report["name"], "revision": report["revision"], "signature": report["model_signature"]}, indent=2), flush=True)
        if args.command in ("search", "run"):
            from .retrieval import search

            directory = search(config, args.activity, offline=args.offline, split=args.split)
        if args.command in ("visualize", "run"):
            from .gallery import visualize

            visualize(config, args.run_dir if args.command == "visualize" else directory, copy_videos=args.copy_videos)
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Re-run the command to reuse completed caches.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"FAILED [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 1
