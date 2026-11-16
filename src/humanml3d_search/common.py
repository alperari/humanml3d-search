"""Configuration, stable input reads and atomic project outputs."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import tempfile

import yaml


class SearchError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise SearchError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def fingerprint(path):
    path = Path(path)
    before = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), f"Input changed while hashing: {path}")
    return {"path": str(path.resolve()), "bytes": after.st_size, "sha256": sha.hexdigest()}


def read_bytes(path):
    path = Path(path)
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), f"Input changed while reading: {path}")
    return content, {"path": str(path.resolve()), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def read_json(path):
    data, facts = read_bytes(path)
    return json.loads(data), facts


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".json-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def now():
    return datetime.now(timezone.utc).isoformat()


def base_id(motion_id):
    require(isinstance(motion_id, str) and re.fullmatch(r"M?\d{6}", motion_id) is not None, f"Invalid HumanML3D motion ID: {motion_id}")
    return motion_id[1:] if motion_id.startswith("M") else motion_id


@dataclass(frozen=True)
class Config:
    dataset_root: Path
    cache_root: Path
    output_root: Path
    activities_file: Path
    model: dict
    index_workers: int
    top_k: int
    source: Path

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        data = yaml.safe_load(path.read_text())
        keys = {"dataset_root", "cache_root", "output_root", "activities_file", "model", "index_workers", "top_k"}
        require(isinstance(data, dict) and set(data) == keys, f"Configuration must contain exactly: {sorted(keys)}")
        for key in ("dataset_root", "cache_root", "output_root", "activities_file"):
            require(isinstance(data[key], str) and data[key], f"Expected a path for {key}")
            data[key] = (path.parent / Path(data[key]).expanduser()).resolve()
        model = data["model"]
        require(isinstance(model, dict) and set(model) == {"name", "revision", "device", "batch_size", "threads"}, "Unsupported model configuration")
        require(model["name"] == "sentence-transformers/all-mpnet-base-v2" and re.fullmatch(r"[0-9a-f]{40}", model["revision"]) is not None,
                "Use all-mpnet-base-v2 with an exact 40-character model commit")
        require(model["device"] in ("cpu", "cuda"), "Model device must be cpu or cuda")
        for label, value in (("index_workers", data["index_workers"]), ("top_k", data["top_k"]),
                             ("batch_size", model["batch_size"]), ("threads", model["threads"])):
            require(type(value) is int and value > 0, f"{label} must be a positive integer")
        result = cls(**data, source=path)
        result.protect(result.cache_root)
        result.protect(result.output_root)
        require(not overlaps(result.cache_root, result.output_root), "Cache and result directories must be separate")
        require(result.dataset_root.is_dir(), f"Dataset directory is missing: {result.dataset_root}")
        return result

    def protect(self, path):
        resolved = Path(path).expanduser().resolve()
        require(not overlaps(resolved, self.dataset_root), f"Output overlaps the original dataset: {resolved}")
        return resolved

    def activities(self):
        content, facts = read_bytes(self.activities_file)
        data = yaml.safe_load(content)
        require(isinstance(data, dict) and set(data) == {"activities"} and isinstance(data["activities"], dict) and data["activities"], "Expected a nonempty activities mapping")
        for name, activity in data["activities"].items():
            require(re.fullmatch(r"[a-z][a-z0-9_]*", name) is not None, "Activity IDs must be safe lowercase identifiers")
            require(set(activity) == {"label", "aliases", "context_rules", "semantic_queries"}, f"Invalid activity fields: {name}")
            for key in ("aliases", "semantic_queries"):
                require(isinstance(activity[key], list) and activity[key] and all(isinstance(v, str) and v.strip() for v in activity[key]), f"Expected {key} strings for {name}")
            require(isinstance(activity["label"], str) and activity["label"].strip(), f"Missing label: {name}")
            for rule in activity["context_rules"]:
                require(set(rule) == {"id", "all"} and isinstance(rule["id"], str) and rule["all"], f"Invalid contextual rule: {name}")
                for expression in rule["all"]:
                    re.compile(expression, re.I)
        return data["activities"], facts


def overlaps(a, b):
    return a == b or a in b.parents or b in a.parents


def environment():
    packages = {}
    for name in ("numpy", "scipy", "torch", "sentence-transformers", "transformers", "huggingface-hub", "tokenizers", "safetensors", "PyYAML", "matplotlib"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def code_facts():
    return {path.name: fingerprint(path)["sha256"] for path in sorted(Path(__file__).parent.glob("*.py"))}
