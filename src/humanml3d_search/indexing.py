"""Immutable caption snapshots with original text, split and segment provenance."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile

import numpy as np

from .common import SearchError, atomic_json, base_id, code_facts, digest, environment, fingerprint, now, read_bytes, read_json, require


def array_header(path, tail):
    before = path.stat()
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        require(version in ((1, 0), (2, 0)), f"Unsupported NPY header at {path}")
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
        offset = stream.tell()
        stream.seek(0)
        header_sha = hashlib.sha256(stream.read(offset)).hexdigest()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), f"Array changed during header read: {path}")
    squeezed = shape == tail
    normalized_shape = (1,) + shape if squeezed else shape
    require(len(normalized_shape) == len(tail) + 1 and normalized_shape[1:] == tail and normalized_shape[0] > 0 and dtype.kind == "f",
            f"Unexpected array shape/type at {path}: {shape}, {dtype}")
    require(after.st_size == offset + math.prod(shape) * dtype.itemsize, f"Array storage size disagrees with header: {path}")
    return {"frames": int(normalized_shape[0]), "shape": list(shape), "singleton_time_squeezed": squeezed, "dtype": str(dtype), "fortran_order": bool(fortran),
            "header_sha256": header_sha, "bytes": after.st_size}


def segment(start_text, end_text, frames):
    warnings = []
    start, end = float(start_text), float(end_text)
    if math.isnan(start):
        start = 0.
        warnings.append("nan_start_to_zero")
    if math.isnan(end):
        end = 0.
        warnings.append("nan_end_to_zero")
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < 0:
        return None, None, None, None, False, "invalid", warnings + ["invalid_time"]
    full = start == end == 0
    if frames is None:
        return start, end, None, None, full, "unpaired", warnings
    if full:
        return start, end, 0, frames, True, "full_clip", warnings
    first, stop = int(start * 20), int(end * 20)
    if end <= start or first >= frames or stop <= first:
        return start, end, None, None, False, "invalid", warnings + ["empty_or_invalid_segment"]
    if stop > frames:
        stop = frames
        warnings.append("segment_end_clipped")
    return start, end, first, stop, False, "clipped" if warnings and "segment_end_clipped" in warnings else "segment", warnings


def inventory(config):
    splits, facts = {}, {}
    for name in ("all", "train", "val", "test", "train_val"):
        content, fact = read_bytes(config.dataset_root / f"{name}.txt")
        ids = content.decode("utf-8").splitlines()
        require(ids and len(ids) == len(set(ids)), f"Empty or duplicate IDs in {name}.txt")
        for value in ids:
            base_id(value)
        splits[name], facts[name] = set(ids), fact
    require(not (splits["train"] & splits["val"] or splits["train"] & splits["test"] or splits["val"] & splits["test"]), "Dataset splits overlap")
    require(splits["all"] == splits["train"] | splits["val"] | splits["test"] and splits["train_val"] == splits["train"] | splits["val"], "Split unions disagree")
    labels = {motion: split for split in ("train", "val", "test") for motion in splits[split]}
    for motion, split in labels.items():
        require(labels.get(("" if motion.startswith("M") else "M") + base_id(motion)) == split, f"Original/mirrored pair is missing or crosses splits: {motion}")
    paths = {}
    for name, suffix in (("texts", ".txt"), ("new_joints", ".npy"), ("new_joint_vecs", ".npy"), ("animations", ".mp4")):
        with os.scandir(config.dataset_root / name) as entries:
            paths[name] = {Path(entry.name).stem: Path(entry.path) for entry in entries if entry.name.endswith(suffix) and entry.is_file()}
    require(splits["all"].issubset(paths["texts"]), "Some split IDs have no caption file")
    return splits, labels, facts, paths


def build_index(config, refresh=False):
    report_path = config.cache_root / "index.json"
    if report_path.exists() and not refresh:
        connection, report = open_index(config)
        connection.close()
        print(f"Using caption snapshot {report['index_signature'][:12]}; use --refresh to re-read dataset inputs", flush=True)
        return report
    config.cache_root.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "stage": "caption_index", "created_at_utc": now(), "dataset_root": str(config.dataset_root),
              "caption_field": "Original first # field only; POS tokens are not indexed", "fps": 20,
              "nan_policy": "Normalize NaN tags to zero as in the upstream loader; retain raw tags and per-caption warnings",
              "motion_hash_policy": "Full caption SHA256; joint/feature header SHA256 and byte sizes. Motion payloads are not hashed at indexing.",
              "environment": environment(), "code_sha256": code_facts(), "status": "running", "outputs": None}
    temporary = None
    connection = None
    try:
        splits, labels, facts, paths = inventory(config)
        report["split_inputs"] = facts
        report["inventory"] = {name: len(values) for name, values in paths.items()}
        report["split_counts"] = {name: len(values) for name, values in splits.items()}
        with tempfile.NamedTemporaryFile(dir=config.cache_root, prefix=".index-", suffix=".sqlite", delete=False) as stream:
            temporary = Path(stream.name)
        connection = sqlite3.connect(temporary)
        connection.executescript("""
            CREATE TABLE motions (motion_id TEXT PRIMARY KEY, base_motion_id TEXT NOT NULL, is_mirrored INTEGER NOT NULL,
                split TEXT, eligible INTEGER NOT NULL, exclusion_reason TEXT, frames INTEGER, has_animation INTEGER NOT NULL, facts_json TEXT NOT NULL);
            CREATE TABLE captions (caption_id TEXT PRIMARY KEY, motion_id TEXT NOT NULL, line_number INTEGER NOT NULL, caption TEXT NOT NULL,
                raw_start TEXT, raw_end TEXT, start_seconds REAL, end_seconds REAL, start_frame INTEGER, end_frame INTEGER,
                full_clip INTEGER NOT NULL, segment_status TEXT NOT NULL, warnings_json TEXT NOT NULL);
            CREATE INDEX caption_motion ON captions(motion_id);
            CREATE TABLE metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        counts, warning_counts, examples, input_digest = Counter(), Counter(), defaultdict(list), hashlib.sha256()
        def process(item):
            motion, path = item
            base = base_id(motion)
            content, caption_facts = read_bytes(path)
            paired = motion in paths["new_joints"] and motion in paths["new_joint_vecs"]
            eligible = motion in splits["all"] and paired
            reason = None if eligible else "not_in_all.txt" if motion not in splits["all"] else "missing_paired_arrays"
            assets = {"caption": {k: caption_facts[k] for k in ("bytes", "sha256")}}
            frames = None
            if paired:
                assets["joints"] = array_header(paths["new_joints"][motion], (22, 3))
                assets["features"] = array_header(paths["new_joint_vecs"][motion], (263,))
                require(assets["joints"]["frames"] == assets["features"]["frames"], f"Joint/feature lengths disagree: {motion}")
                frames = assets["joints"]["frames"]
            require(motion not in splits["all"] or eligible, f"Paired motion data missing for split ID {motion}")
            rows = []
            for number, line in enumerate(content.decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split("#")
                require(len(parts) == 4 and parts[0].strip(), f"Malformed caption: {motion}, line {number}")
                caption, _, start, end = parts
                timing = segment(start, end, frames)
                rows.append((f"{motion}:{number:04d}", motion, number, caption, start, end, *timing[:4], int(timing[4]), timing[5], json.dumps(timing[6])))
            require(rows, f"No captions in {motion}")
            values = (motion, base, int(motion.startswith("M")), labels.get(motion), int(eligible), reason, frames,
                      int(motion in paths["animations"]), json.dumps(assets, sort_keys=True))
            return values, rows
        items = sorted(paths["texts"].items())
        with ThreadPoolExecutor(max_workers=config.index_workers) as executor:
            for number, (motion, rows) in enumerate(executor.map(process, items), 1):
                connection.execute("INSERT INTO motions VALUES (?,?,?,?,?,?,?,?,?)", motion)
                connection.executemany("INSERT INTO captions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                input_digest.update(json.dumps(motion, sort_keys=True).encode())
                counts["text_files"] += 1
                counts["caption_rows"] += len(rows)
                counts["eligible_motion_ids"] += motion[4]
                counts["eligible_caption_rows"] += len(rows) if motion[4] else 0
                counts["excluded_motion_ids"] += not motion[4]
                assets = json.loads(motion[8])
                counts["singleton_time_squeezed_motion_ids"] += bool(assets.get("joints", {}).get("singleton_time_squeezed"))
                for row in rows:
                    counts[f"timing_{row[11]}"] += 1
                    for warning in json.loads(row[12]):
                        warning_counts[warning] += 1
                        if len(examples[warning]) < 5:
                            examples[warning].append(row[0])
                if number % 2000 == 0 or number == len(items):
                    print(f"Indexed {number}/{len(items)} caption files ({counts['caption_rows']} captions)", flush=True)
        counts["eligible_base_motion_ids"] = connection.execute("SELECT COUNT(DISTINCT base_motion_id) FROM motions WHERE eligible=1").fetchone()[0]
        report.update(counts=dict(counts), warning_counts=dict(warning_counts), warning_examples=dict(examples),
                      excluded_motions=[dict(zip(("motion_id", "reason"), row)) for row in connection.execute("SELECT motion_id,exclusion_reason FROM motions WHERE eligible=0 ORDER BY motion_id")])
        identity = {"schema_version": 1, "dataset_root": str(config.dataset_root), "inputs_sha256": input_digest.hexdigest(),
                    "split_sha256": {name: fact["sha256"] for name, fact in facts.items()}, "counts": dict(counts), "index_code_sha256": report["code_sha256"]["indexing.py"]}
        report["index_identity"], report["index_signature"] = identity, digest(identity)
        connection.execute("INSERT INTO metadata VALUES (?,?)", ("index_signature", report["index_signature"]))
        connection.commit()
        connection.close()
        connection = None
        archive = config.cache_root / "caption_index.sqlite"
        os.replace(temporary, archive)
        report["status"], report["outputs"] = "passed", {"index": fingerprint(archive)}
    except Exception as exc:
        report["status"], report["error"] = "failed", str(exc)
        raise
    finally:
        if connection is not None:
            connection.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        atomic_json(report_path, report)
    return report


def open_index(config):
    report, _ = read_json(config.cache_root / "index.json")
    require(report.get("status") == "passed" and report.get("schema_version") == 1 and report.get("dataset_root") == str(config.dataset_root), "Build a passed index for this dataset first")
    path = config.cache_root / "caption_index.sqlite"
    expected = report["outputs"]["index"]
    actual = fingerprint(path)
    require(actual["sha256"] == expected["sha256"] and actual["bytes"] == expected["bytes"], "Caption index no longer matches its manifest; rebuild with index --refresh")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    require(connection.execute("SELECT value FROM metadata WHERE name='index_signature'").fetchone()[0] == report["index_signature"], "Index signature disagrees")
    return connection, report


def caption_rows(config):
    connection, report = open_index(config)
    try:
        rows = [dict(row) for row in connection.execute("""SELECT c.*,m.base_motion_id,m.is_mirrored,m.split,m.frames,m.has_animation
            FROM captions c JOIN motions m USING(motion_id) WHERE m.eligible=1 ORDER BY c.motion_id,c.line_number""")]
    finally:
        connection.close()
    for row in rows:
        row["is_mirrored"], row["full_clip"] = bool(row["is_mirrored"]), bool(row["full_clip"])
        row["warnings"] = json.loads(row.pop("warnings_json"))
    return rows, report
