"""Lexical matching, exact cosine ranking and overlap comparison of motions."""

from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata

import numpy as np

from .common import atomic_json, code_facts, digest, environment, fingerprint, now, require
from .embeddings import prepare_embeddings
from .indexing import caption_rows

COMMON_COLUMNS = ["exercise_id", "motion_id", "base_motion_id", "is_mirrored", "split"]
CAPTION_COLUMNS = ["caption_id", "caption", "start_seconds", "end_seconds", "start_frame", "end_frame", "segment_status"]
LEXICAL_COLUMNS = COMMON_COLUMNS + ["match_type", "matched_rules", "matched_terms", "matched_caption_count"] + CAPTION_COLUMNS
SEMANTIC_COLUMNS = COMMON_COLUMNS + ["rank", "cosine_similarity", "query_id", "query", "caption_count"] + CAPTION_COLUMNS


def lexical_matches(rows, activity_id, activity):
    patterns = []
    for alias in activity["aliases"]:
        words = re.findall(r"\w+", unicodedata.normalize("NFKC", alias).casefold())
        require(bool(words), f"Empty lexical alias: {alias}")
        expression = r"(?<!\w)" + r"[\s-]+".join(re.escape(word) for word in words) + r"s?(?!\w)"
        patterns.append((f"alias:{alias}", re.compile(expression, re.I)))
    contexts = [(rule["id"], [re.compile(expression, re.I) for expression in rule["all"]]) for rule in activity["context_rules"]]
    hits = defaultdict(list)
    for row in rows:
        text = unicodedata.normalize("NFKC", row["caption"]).casefold().translate(str.maketrans({c: "-" for c in "‐‑‒–—−"}))
        explicit = [(name, match.group()) for name, pattern in patterns for match in pattern.finditer(text)]
        contextual = [name for name, expressions in contexts if all(expression.search(text) for expression in expressions)]
        if not explicit and not contextual:
            continue
        hits[row["motion_id"]].append(dict(row, exercise_id=activity_id, match_type="explicit" if explicit else "contextual",
                                           matched_rules=sorted({name for name, _ in explicit} | {f"context:{name}" for name in contextual}),
                                           matched_terms=sorted({term for _, term in explicit})))
    results, evidence = [], []
    for motion_id, matches in sorted(hits.items()):
        matches.sort(key=lambda r: (r["match_type"] != "explicit", r["line_number"]))
        best = dict(matches[0], matched_caption_count=len(matches))
        results.append({key: best[key] for key in LEXICAL_COLUMNS})
        evidence.extend(dict(match, method="lexical") for match in matches)
    return results, evidence


def semantic_matches(config, rows, activity_id, activity, model, matrix, text_index):
    from threadpoolctl import threadpool_limits

    queries = model.encode(activity["semantic_queries"], batch_size=config.model["batch_size"], normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False, precision="float32")
    require(queries.shape == (len(activity["semantic_queries"]), 768) and np.isfinite(queries).all(), "Invalid query embeddings")
    with threadpool_limits(limits=config.model["threads"]):
        scores = np.asarray(matrix @ queries.T)
    require(np.isfinite(scores).all(), "Cosine similarities contain nonfinite values")
    winners, caption_counts = {}, defaultdict(int)
    for row in rows:
        values = scores[text_index[row["caption"]]]
        query_number = int(np.argmax(values))
        candidate = dict(row, exercise_id=activity_id, cosine_similarity=float(values[query_number]),
                         query_id=query_number + 1, query=activity["semantic_queries"][query_number])
        caption_counts[row["motion_id"]] += 1
        current = winners.get(row["motion_id"])
        if current is None or candidate["cosine_similarity"] > current["cosine_similarity"]:
            winners[row["motion_id"]] = candidate
    ranked = sorted(winners.values(), key=lambda r: (-r["cosine_similarity"], r["motion_id"]))
    require(len(ranked) >= config.top_k, f"Only {len(ranked)} eligible motions; cannot return {config.top_k}")
    for rank, row in enumerate(ranked, 1):
        row.update(rank=rank, caption_count=caption_counts[row["motion_id"]])
    results = [{key: row[key] for key in SEMANTIC_COLUMNS} for row in ranked[:config.top_k]]
    selected = {row["motion_id"] for row in results}
    evidence = []
    for row in rows:
        if row["motion_id"] in selected:
            values = scores[text_index[row["caption"]]]
            query_number = int(np.argmax(values))
            evidence.append(dict(row, exercise_id=activity_id, method="semantic", cosine_similarity=float(values[query_number]),
                                 query_id=query_number + 1, query=activity["semantic_queries"][query_number], query_similarities=values.tolist()))
    return results, evidence


def result_set(activity_id, activity, method, rows):
    ids = {row["motion_id"] for row in rows}
    require(len(ids) == len(rows), f"{method} retrieval contains duplicate motion IDs")
    return {"schema_version": 2, "exercise_id": activity_id, "label": activity["label"], "method": method, "records": rows,
            "counts": {"motion_ids": len(ids), "base_motion_ids": len({row["base_motion_id"] for row in rows})},
            "count_interpretation": "Base IDs group mirrored counterparts within this method, not proven distinct people or raw recordings"}


def overlap_results(activity_id, activity, lexical, semantic):
    left, right = {row["motion_id"]: row for row in lexical}, {row["motion_id"]: row for row in semantic}
    shared_ids = left.keys() & right.keys()
    records = [dict({key: right[motion][key] for key in COMMON_COLUMNS}, lexical=left[motion], semantic=right[motion])
               for motion in sorted(shared_ids, key=lambda motion: right[motion]["rank"])]
    groups = {}
    for method, rows in (("lexical", lexical), ("semantic", semantic)):
        groups[method] = defaultdict(list)
        for row in rows:
            groups[method][row["base_motion_id"]].append(row["motion_id"])
    shared_bases = groups["lexical"].keys() & groups["semantic"].keys()
    return {"schema_version": 2, "exercise_id": activity_id, "label": activity["label"], "method": "overlap", "records": records,
            "counts": {"motion_ids": len(shared_ids), "base_motion_ids": len(shared_bases)},
            "exact_match_base_motion_ids": len({row["base_motion_id"] for row in records}),
            "mirror_overlap_groups": [{"base_motion_id": base, "lexical_motion_ids": sorted(groups["lexical"][base]),
                                       "semantic_motion_ids": sorted(groups["semantic"][base])} for base in sorted(shared_bases)],
            "count_interpretation": "Records are exact motion-ID matches. Base-ID overlap is computed after grouping mirrors separately in each method; a base-only match need not share an exact ID."}


def write_csv(path, rows, columns):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=".csv-", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value for key, value in row.items()})
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_jsonl(path, rows):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".evidence-", delete=False) as stream:
            temporary = Path(stream.name)
            for row in sorted(rows, key=lambda r: (r["motion_id"], r["line_number"])):
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def search(config, activity_id, offline=False, split="all"):
    activities, activity_facts = config.activities()
    require(activity_id in activities, f"Unknown activity {activity_id}; choices: {', '.join(activities)}")
    require(split in ("all", "train", "val", "test"), "Unsupported split")
    all_rows, index_report = caption_rows(config)
    model, matrix, text_index, embedding_report = prepare_embeddings(config, all_rows, offline)
    rows = all_rows if split == "all" else [row for row in all_rows if row["split"] == split]
    activity = activities[activity_id]
    lexical, lexical_evidence = lexical_matches(rows, activity_id, activity)
    semantic, semantic_evidence = semantic_matches(config, rows, activity_id, activity, model, matrix, text_index)
    results = {"lexical": result_set(activity_id, activity, "lexical", lexical),
               "semantic": result_set(activity_id, activity, "semantic", semantic)}
    overlap = overlap_results(activity_id, activity, lexical, semantic)
    settings = {"index_signature": index_report["index_signature"], "embedding_signature": embedding_report["embedding_signature"],
                "exercise_id": activity_id, "activity": activity, "top_k": config.top_k, "split": split,
                "ranking": "Maximum normalized cosine over activity queries and each motion's original captions",
                "tie_break": "Descending score, ascending motion ID; first caption line/query wins exact ties",
                "semantic_unit": "Distinct motion IDs; mirrored IDs can occupy separate top-k positions",
                "result_schema_version": 2, "code_sha256": code_facts()}
    run_signature = digest(settings)
    directory = config.protect(config.output_root / activity_id / run_signature[:16])
    directory.mkdir(parents=True, exist_ok=True)
    counts = {method: value["counts"] for method, value in results.items()}
    counts["overlap"] = overlap["counts"]
    report = {"schema_version": 2, "stage": "caption_search", "status": "running", "created_at_utc": now(), "run_signature": run_signature,
              "settings": settings, "dataset_root": str(config.dataset_root), "counts": counts,
              "environment": environment(), "inputs": {"index_manifest": fingerprint(config.cache_root / "index.json"),
                                                         "activities": activity_facts, "embedding_manifest": fingerprint(Path(embedding_report["vectors"]["path"]).parent / "manifest.json")},
              "model": embedding_report["model"], "outputs": None, "visual_review": "pending"}
    atomic_json(directory / "run_manifest.json", report)
    try:
        write_csv(directory / "lexical_results.csv", lexical, LEXICAL_COLUMNS)
        write_csv(directory / "semantic_results.csv", semantic, SEMANTIC_COLUMNS)
        for method, evidence in (("lexical", lexical_evidence), ("semantic", semantic_evidence)):
            atomic_json(directory / f"{method}_results.json", results[method])
            write_jsonl(directory / f"{method}_matches.jsonl", evidence)
        atomic_json(directory / "overlap_results.json", overlap)
        report["outputs"] = {name: fingerprint(directory / name) for method in results for name in
                             (f"{method}_results.csv", f"{method}_results.json", f"{method}_matches.jsonl")}
        report["outputs"]["overlap_results.json"] = fingerprint(directory / "overlap_results.json")
        report["status"] = "passed"
    except Exception as exc:
        report["status"], report["error"] = "failed", str(exc)
        raise
    finally:
        atomic_json(directory / "run_manifest.json", report)
    print(json.dumps({"run_directory": str(directory), "counts": counts}, indent=2), flush=True)
    return directory
