"""Pinned local sentence embeddings, cached independently of activity queries."""

from pathlib import Path
import os
import tempfile
import time

import numpy as np

from .common import atomic_json, digest, environment, fingerprint, now, read_json, require

MODEL_FILES = ["config.json", "config_sentence_transformers.json", "model.safetensors", "modules.json", "sentence_bert_config.json",
               "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "1_Pooling/config.json"]


def fetch_model(config, offline=False):
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=config.model["name"], revision=config.model["revision"],
                                     cache_dir=config.cache_root / "models", allow_patterns=MODEL_FILES, local_files_only=offline))
    require(snapshot.name == config.model["revision"], "Model snapshot does not match the configured commit")
    require((snapshot / "model.safetensors").is_file(), "Pinned safetensors weights are missing; run model without --offline once")
    facts = {str(path.relative_to(snapshot)): {key: value for key, value in fingerprint(path).items() if key != "path"}
             for path in sorted(snapshot.rglob("*")) if path.is_file()}
    require(set(facts).issubset(MODEL_FILES), "Unexpected file in the model snapshot")
    identity = {"name": config.model["name"], "revision": config.model["revision"], "files": facts}
    report = dict(identity, model_signature=digest(identity), snapshot_path=str(snapshot), created_at_utc=now())
    atomic_json(config.cache_root / "model.json", report)
    return report


def load_model(config, offline=False):
    import torch
    from sentence_transformers import SentenceTransformer

    report = fetch_model(config, offline)
    require(config.model["device"] != "cuda" or torch.cuda.is_available(), "CUDA was selected but is unavailable in this environment")
    torch.set_num_threads(config.model["threads"])
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = SentenceTransformer(report["snapshot_path"], device=config.model["device"], local_files_only=True,
                                trust_remote_code=False, model_kwargs={"dtype": torch.float32})
    model.eval()
    require(model.get_sentence_embedding_dimension() == 768, "Expected all-mpnet-base-v2's 768-dimensional output")
    return model, report


def prepare_embeddings(config, rows, offline=False):
    model, model_report = load_model(config, offline)
    texts = sorted({row["caption"] for row in rows})
    versions = environment()["packages"]
    identity = {"model_signature": model_report["model_signature"], "unique_captions_sha256": digest(texts),
                "unique_captions": len(texts), "dimensions": 768, "dtype": "float32", "normalize_embeddings": True,
                "max_seq_length": model.max_seq_length, "device": config.model["device"], "threads": config.model["threads"],
                "batch_size": config.model["batch_size"], "seed": 0, "deterministic_algorithms": True,
                "versions": {key: versions[key] for key in ("numpy", "torch", "sentence-transformers", "transformers", "tokenizers", "safetensors")}}
    key = digest(identity)
    directory = config.cache_root / "embeddings" / key
    directory.mkdir(parents=True, exist_ok=True)
    report_path, array_path = directory / "manifest.json", directory / "captions.npy"
    if report_path.exists():
        report, _ = read_json(report_path)
        require(report["identity"] == identity and report["status"] == "passed", "Embedding cache identity differs")
        require(fingerprint(array_path)["sha256"] == report["vectors"]["sha256"], "Embedding cache checksum mismatch")
        print(f"Using cached embeddings for {len(texts)} unique natural-language captions", flush=True)
    else:
        temporary = None
        started = time.monotonic()
        try:
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".vectors-", suffix=".npy", delete=False) as stream:
                temporary = Path(stream.name)
            matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(texts), 768))
            for start in range(0, len(texts), 512):
                stop = min(start + 512, len(texts))
                values = model.encode(texts[start:stop], batch_size=config.model["batch_size"], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False, precision="float32")
                require(values.shape == (stop - start, 768) and np.isfinite(values).all(), "Invalid caption embeddings")
                require(np.max(np.abs(np.linalg.norm(values, axis=1) - 1)) < 1e-5, "Caption embeddings are not unit normalized")
                matrix[start:stop] = values
                elapsed = time.monotonic() - started
                print(f"Embedded {stop}/{len(texts)} unique captions; {elapsed:.1f}s elapsed", flush=True)
            matrix.flush()
            del matrix
            os.replace(temporary, array_path)
            report = {"status": "passed", "created_at_utc": now(), "embedding_signature": key, "identity": identity,
                      "vectors": fingerprint(array_path), "model": model_report, "elapsed_seconds": time.monotonic() - started}
            atomic_json(report_path, report)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    matrix = np.load(array_path, mmap_mode="r", allow_pickle=False)
    require(matrix.shape == (len(texts), 768) and matrix.dtype == np.float32, "Cached vector shape/type mismatch")
    return model, matrix, {text: number for number, text in enumerate(texts)}, report
