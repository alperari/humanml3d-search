"""Lexical matching, exact cosine ranking, and mirror-aware motion unions."""

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
    return results, evidence, {row["motion_id"]: {key: row[key] for key in SEMANTIC_COLUMNS} for row in ranked}


def combine(activity_id, activity, lexical, semantic, ranking):
    left, right = {r["motion_id"]: r for r in lexical}, {r["motion_id"]: r for r in semantic}
    require(len(left) == len(lexical) and len(right) == len(semantic), "Retrieval contains duplicate motion IDs")
    ids = set(left) | set(right)
    records, groups = [], defaultdict(list)
    for motion_id in sorted(ids, key=lambda m: ranking[m]["rank"]):
        sample = ranking[motion_id]
        record = {key: sample[key] for key in COMMON_COLUMNS}
        record.update(methods=[method for method, source in (("lexical", left), ("semantic", right)) if motion_id in source],
                      lexical=left.get(motion_id), semantic=right.get(motion_id), best_semantic_caption=sample)
        records.append(record)
        groups[sample["base_motion_id"]].append(motion_id)
    def counts(a, b):
        return {"lexical": len(a), "semantic": len(b), "overlap": len(a & b), "lexical_only": len(a - b), "semantic_only": len(b - a), "union": len(a | b)}
    mirror_groups = [{"base_motion_id": base, "motion_ids": sorted(values),
                      "representative_motion_id": base if base in values else sorted(values)[0]}
                     for base, values in sorted(groups.items())]
    return {"schema_version": 1, "exercise_id": activity_id, "label": activity["label"], "records": records, "mirror_groups": mirror_groups,
            "counts": {"motion_ids": counts(set(left), set(right)),
                       "mirror_adjusted": counts({r["base_motion_id"] for r in lexical}, {r["base_motion_id"] for r in semantic})},
            "count_interpretation": "Mirror-adjusted counts are distinct original HumanML3D IDs, not proven distinct people or raw recordings"}


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


def search(config, activity_id, offline=False, split="all"):
    activities, activity_facts = config.activities()
    require(activity_id in activities, f"Unknown activity {activity_id}; choices: {', '.join(activities)}")
    require(split in ("all", "train", "val", "test"), "Unsupported split")
    all_rows, index_report = caption_rows(config)
    model, matrix, text_index, embedding_report = prepare_embeddings(config, all_rows, offline)
    rows = all_rows if split == "all" else [row for row in all_rows if row["split"] == split]
    activity = activities[activity_id]
    lexical, lexical_evidence = lexical_matches(rows, activity_id, activity)
    semantic, semantic_evidence, ranking = semantic_matches(config, rows, activity_id, activity, model, matrix, text_index)
    combined = combine(activity_id, activity, lexical, semantic, ranking)
    settings = {"index_signature": index_report["index_signature"], "embedding_signature": embedding_report["embedding_signature"],
                "exercise_id": activity_id, "activity": activity, "top_k": config.top_k, "split": split,
                "ranking": "Maximum normalized cosine over activity queries and each motion's original captions",
                "tie_break": "Descending score, ascending motion ID; first caption line/query wins exact ties",
                "semantic_unit": "Distinct motion IDs; mirrored IDs can occupy separate top-k positions", "code_sha256": code_facts()}
    run_signature = digest(settings)
    directory = config.protect(config.output_root / activity_id / run_signature[:16])
    directory.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "stage": "caption_search", "status": "running", "created_at_utc": now(), "run_signature": run_signature,
              "settings": settings, "dataset_root": str(config.dataset_root), "counts": combined["counts"],
              "environment": environment(), "inputs": {"index_manifest": fingerprint(config.cache_root / "index.json"),
                                                         "activities": activity_facts, "embedding_manifest": fingerprint(Path(embedding_report["vectors"]["path"]).parent / "manifest.json")},
              "model": embedding_report["model"], "outputs": None, "visual_review": "pending"}
    atomic_json(directory / "run_manifest.json", report)
    try:
        write_csv(directory / "lexical_results.csv", lexical, LEXICAL_COLUMNS)
        write_csv(directory / "semantic_results.csv", semantic, SEMANTIC_COLUMNS)
        atomic_json(directory / "combined_results.json", combined)
        evidence = sorted(lexical_evidence + semantic_evidence, key=lambda r: (r["method"], r["motion_id"], r["line_number"]))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, prefix=".evidence-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                for row in evidence:
                    stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, directory / "caption_matches.jsonl")
        report["outputs"] = {name: fingerprint(directory / name) for name in
                             ("lexical_results.csv", "semantic_results.csv", "combined_results.json", "caption_matches.jsonl")}
        report["status"] = "passed"
    except Exception as exc:
        report["status"], report["error"] = "failed", str(exc)
        raise
    finally:
        atomic_json(directory / "run_manifest.json", report)
    print(json.dumps({"run_directory": str(directory), "counts": combined["counts"]}, indent=2), flush=True)
    return directory
