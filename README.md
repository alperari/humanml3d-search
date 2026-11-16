# HumanML3D activity search

Lexical and semantic search over the original HumanML3D natural-language captions. The initial activity is **push-up**. Both searches use the first field of each caption line, never the POS-tagged tokens. The original dataset remains read-only.

For each configured activity, the workflow writes separate `lexical_results.csv` and `semantic_results.csv` files, reports their overlap, and creates lexical, semantic and overlap video galleries. Each selection includes counts with mirrored counterparts grouped together. Retrieved motions are candidates for human review, rather than automatically accepted exercise examples.

## Environment

Run commands from this repository's root. The supplied environment targets Linux x86-64 with CPU inference; it is independent of the Fit3D conversion environment.

```bash
cd /home/aari/projects-thesis/humanml3d-search
conda env create -f environment.yml
conda activate humanml3d-search
```

For the exact conda package builds used for the recorded run, recreate from both lock files instead:

```bash
conda create -n humanml3d-search --file conda-linux-64.lock.txt
conda activate humanml3d-search
python -m pip install -r requirements-linux-cpu.lock.txt
python -m pip install --no-deps --no-build-isolation -e .
```

`requirements.in` records the direct dependency choices; `requirements-linux-cpu.lock.txt` pins the installed Python dependency closure. `conda-linux-64.lock.txt` pins the native package URLs. The editable application is installed from this checkout, so retain the source alongside these locks. The ordinary `environment.yml` pins the same Python dependencies but allows the native conda solver to choose compatible builds. A GPL FFmpeg build with `libx264` is included for fallback rendering.

The model is `sentence-transformers/all-mpnet-base-v2`, pinned to commit `e8c3b32edf5434bc2275fc9bab85f82640a19130`. The first model download needs internet access. Caption embedding and search run locally; dataset captions are not sent to an embedding service. Once the model is cached, use `--offline`.

## Configuration

`configs/local.yml` points to `/home/aari/data_share/HumanML3D`. To use another installation, copy `configs/example.yml` to `configs/user.yml` and edit `dataset_root`. Relative paths are resolved against the config file's directory. Every command accepts `--config PATH`.

Expected dataset inputs:

```text
HumanML3D/
  all.txt  train.txt  val.txt  test.txt  train_val.txt
  texts/<motion_id>.txt
  new_joints/<motion_id>.npy
  new_joint_vecs/<motion_id>.npy
  animations/<motion_id>.mp4
```

The arrays and official split files establish eligible motions; search does not reprocess or normalize the motion features. Existing animations are reused. If an individual preview is absent, the gallery renders the corresponding `new_joints` array at 20 fps with HumanML3D's 22-joint chains. The fallback uses a root-following Matplotlib view; it does not require the Fit3D project or a separate HumanML3D source checkout.

## Run progressively

1. Build the caption and motion-availability snapshot:

   ```bash
   humanml3d-search index --config configs/local.yml
   ```

   Inspect `cache/index.json` for counts, exclusions, segment warnings and array shapes. The complete indexed rows and per-motion input fingerprints are in `cache/caption_index.sqlite`.

2. Download and fingerprint the pinned model:

   ```bash
   humanml3d-search model --config configs/local.yml
   ```

   Inspect `cache/model.json` for the commit and SHA256 of each downloaded model/tokenizer file. This step downloads the files; the search step loads the model and computes embeddings.

3. Run both search methods and report their overlap:

   ```bash
   humanml3d-search search --config configs/local.yml --activity push_up --offline
   ```

   The command prints the result directory and both sets of counts. The first search embeds the unique natural-language captions; subsequent activities reuse those corpus embeddings. `--split train`, `--split val` and `--split test` restrict retrieval to one official split. The default is `--split all`.

4. Build the review galleries for that result directory:

   ```bash
   humanml3d-search visualize --config configs/local.yml \
     --run-dir outputs/push_up/<run_signature>
   ```

   Open `lexical_review.html`, `semantic_review.html` or `overlap_review.html` in a browser. Each gallery groups original/mirrored pairs by default, supports a text filter, and provides full-motion and matched-caption playback. The overlap gallery contains exact motion-ID matches and shows each search's representative caption. Use `--method lexical`, `--method semantic` or `--method overlap` to build just one gallery. Use `--copy-videos` when the galleries must be portable; otherwise `media/<selection>/` contains links to the original previews. For a remote machine, serve the result directory locally and forward the port:

   ```bash
   python -m http.server 8000 --bind 127.0.0.1 --directory outputs/push_up/<run_signature>
   ```

   Then open `http://127.0.0.1:8000/lexical_review.html`, `http://127.0.0.1:8000/semantic_review.html` or `http://127.0.0.1:8000/overlap_review.html` on the machine with that port available.

After reviewing the stages, the equivalent single command is:

```bash
humanml3d-search run --config configs/local.yml --activity push_up --offline
```

Omit `--offline` on a fresh installation to allow the model download. All commands also work as `python -m humanml3d_search ...`.

The default search covers **train, validation and test**. To retrieve training motions only, add `--split train` to `search` or `run`; semantic ranking then selects the top 50 within that split.

## Search definitions

**Lexical:** configurable activity aliases and contextual regular-expression rules. Push-up aliases cover `push-up`, `push up`, `pushup`, `press-up`, `press up`, and `pressup`, including plurals, case and common Unicode dash variants. Contextual rules require all configured expressions in the same caption. Only matching uses Unicode normalization; stored/output captions retain the original text. Explicit matches take priority over contextual matches when choosing a representative caption, followed by original caption line number. The evidence file retains every matching caption and rule.

**Semantic:** normalized float32, 768-dimensional MPNet embeddings and exact cosine similarity. Each motion's score is the maximum similarity across its original captions and the configured activity queries. The result contains the top **50 distinct motion IDs**, with one winning caption and query per motion. Scores tie-break by motion ID; exact caption/query ties use their original order. A mirrored motion can occupy its own top-50 position. The embedding cache records the model files, corpus text digest, library versions, device, batch size, thread count and deterministic settings. Exact numerical agreement across different hardware or library builds is not guaranteed; use the supplied environment and recorded settings for comparisons.

**Overlap and counts:** retain each search's results independently. `overlap_results.json` lists only exact motion IDs found by both searches, with their lexical and semantic caption evidence. Its mirror-adjusted overlap is calculated by intersecting the two methods' base-ID sets. `base_motion_id` removes exactly one leading `M`, so `000123` and `M000123` belong to the same group. A shared base ID can exist without an exact shared motion ID; these cases remain in `mirror_overlap_groups`, while `exact_match_base_motion_ids` counts groups represented in the exact-ID gallery. Base IDs identify distinct original HumanML3D IDs, not distinct people or underlying raw recordings. Each gallery prefers an original ID when both counterparts occur in that selection.

Captions such as “push up from the floor” or “assume a push-up position” can retrieve a get-up or a static pose. Contextual rules and semantic similarity can also retrieve related actions. Scores are ranking values, not calibrated probabilities. Human motion review is required to determine which clips actually contain the requested exercise.

To add an activity, add another entry to `configs/activities.yml` with `label`, `aliases`, `context_rules`, and `semantic_queries`, then pass its key to `--activity`. Keep every semantic query specific to the exercise: the maximum-over-queries policy allows a broad arm-movement description to dominate the ranking. The entire activities file and the selected entry are recorded in the run provenance. Corpus embeddings are reusable without modification.

## Outputs and provenance

Each run is written to `outputs/<activity>/<run_signature[:16]>/`:

| Artifact | Contents |
| --- | --- |
| `lexical_results.csv` | One row per lexical motion ID: representative original caption, explicit/contextual category, matching rules/terms, matching-caption count and interval. |
| `semantic_results.csv` | One row per top-50 motion ID: rank, cosine score, winning original caption/query, caption count and interval. |
| `lexical_results.json`, `semantic_results.json` | Each method's motion records and raw/mirror-adjusted counts, stored independently. |
| `overlap_results.json` | Exact shared motion IDs with both methods' representative captions, plus mirror-adjusted overlap groups and counts. |
| `lexical_matches.jsonl` | Every lexical caption match, including rules, raw time tags and warnings. |
| `semantic_matches.jsonl` | Every caption of each semantic top-50 motion, including per-query scores, raw time tags and warnings. |
| `run_manifest.json` | Stage status, index/embedding/activity identities, lexical/semantic/overlap counts, runtime versions, scoring policy, source hashes and result hashes. |
| `lexical_review.html`, `semantic_review.html`, `overlap_review.html` | Review galleries for each search and their exact-ID overlap. |
| `media/<selection>/` | Video links or copies for the corresponding gallery. |
| `<selection>_gallery_manifest.json` | Gallery stage status, search-manifest fingerprint, video source fingerprints, codec/rate metadata and fallback-render details where used. |

Both CSVs include `exercise_id`, `motion_id`, `base_motion_id`, `is_mirrored`, and the official `split`. Motion IDs are strings: preserve leading zeros when importing into spreadsheets or dataframe libraries. List-valued CSV fields contain JSON arrays.

The corpus index is an explicit immutable snapshot. Cached commands verify its checksum and identity, but do not reread all live dataset files. After changing dataset contents, run `humanml3d-search index --config configs/local.yml --refresh`. Caption contents and split files receive full SHA256 hashes; the initial motion inventory validates array headers, shapes, byte sizes and paired frame counts, without hashing or scanning every motion payload. Rendered joints and reused preview videos receive full hashes when visualized. Search signatures bind the index, embeddings, selected activity, top-k, split and application source hashes. Completed files are published with atomic replacement; incomplete stages receive a failed manifest when an exception is handled.

## Dataset conventions retained

- Each nonblank text row is `caption#POS_tokens#start_seconds#end_seconds`. The second field is ignored for matching and embedding.
- `0#0` marks the whole motion. Following the upstream text-to-motion loader, NaN tags become zero; raw strings and warnings are retained in the index/evidence.
- Nonzero intervals use half-open frame bounds `[int(start * 20), int(end * 20))`. An end beyond the available frames is clipped and flagged. Invalid/empty intervals remain searchable as captions, with no fabricated segment playback.
- Supplied `new_joints` and `new_joint_vecs` keep their existing alignment and frame counts. Singleton arrays whose time dimension was squeezed out are recognized as one-frame motions; raw shapes are retained in the index.
- All official split IDs must have captions and both arrays. Original/mirrored pairs must share a split. Caption files outside `all.txt` are indexed for audit and excluded from retrieval.

The local inventory contains 29,232 text files / 87,384 captions. Of these, 29,228 motion IDs / 87,372 captions are eligible, covering 14,614 base IDs. Four extra caption files (`009707`, `011059` and their `M` counterparts) have no paired arrays and are excluded. Four motions have squeezed singleton time dimensions. The index records 1,010 clipped intervals and eight invalid intervals, plus NaN normalization warnings.

The completed local push-up run provides a [lexical gallery](outputs/push_up/880932ee6d8c2bd3/lexical_review.html), a [semantic gallery](outputs/push_up/880932ee6d8c2bd3/semantic_review.html) and an [overlap gallery](outputs/push_up/880932ee6d8c2bd3/overlap_review.html). Its all-split counts are:

| Selection | Motion IDs | Mirror-adjusted base IDs |
| --- | ---: | ---: |
| Lexical | 86 | 43 |
| Semantic top 50 | 50 | 25 |
| Overlap | 28 | 14 |

Each gallery reuses the original previews with H.264 encoding and 20 fps. The lexical, semantic and overlap galleries initially show 43, 25 and 14 representatives, respectively. See [the run record](docs/first_run.md) for the completed commands, provenance and query refinement.

Implementation verification uses actual dataset indexing, model inference, search and rendering. No automated test suite is included.

References: [HumanML3D source repository](https://github.com/EricGuo5513/HumanML3D), [upstream text-to-motion caption loader](https://github.com/EricGuo5513/text-to-motion/blob/main/data/dataset.py), [MPNet model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).
