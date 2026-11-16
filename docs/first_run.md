# Recorded local push-up search

The workflow runs in the isolated `humanml3d-search` conda environment against the original dataset at `/home/aari/data_share/HumanML3D`. Lexical and semantic results remain separate; an overlap report and gallery retain motions found by both searches. The source dataset is read-only. No automated tests were added or run.

## Completed workflow

```bash
conda activate humanml3d-search
humanml3d-search run --config configs/local.yml --activity push_up --offline
```

This run covers all official splits. Add `--split train` to search training data only. The dataset snapshot contains 29,232 caption files / 87,384 captions, with 29,228 eligible motion IDs / 87,372 captions representing 14,614 base IDs. The 55,858 unique eligible captions were originally embedded in approximately 398 seconds on CPU with eight threads and batch size 32. This run reused that cache and completed offline.

| Selection | Distinct motion IDs | Mirror-adjusted base IDs |
| --- | ---: | ---: |
| Lexical | 86 | 43 |
| Semantic | 50 | 25 |
| Overlap | 28 | 14 |

The lexical representatives comprise 56 explicit-name matches and 30 contextual matches. The semantic CSV contains ranks 1–50, with cosine scores from approximately 0.8580 to 0.7913. Motion `009006` and its mirrored counterpart lead the ranking with the caption “the person does multiple push-ups”. These are search candidates awaiting human relevance review.

## Result artifacts

- [Lexical CSV](../outputs/push_up/880932ee6d8c2bd3/lexical_results.csv) and [caption evidence](../outputs/push_up/880932ee6d8c2bd3/lexical_matches.jsonl).
- [Semantic CSV](../outputs/push_up/880932ee6d8c2bd3/semantic_results.csv) and [caption evidence](../outputs/push_up/880932ee6d8c2bd3/semantic_matches.jsonl).
- [Overlap records and counts](../outputs/push_up/880932ee6d8c2bd3/overlap_results.json).
- [Lexical gallery](../outputs/push_up/880932ee6d8c2bd3/lexical_review.html), [semantic gallery](../outputs/push_up/880932ee6d8c2bd3/semantic_review.html), and [overlap gallery](../outputs/push_up/880932ee6d8c2bd3/overlap_review.html).
- [Search manifest](../outputs/push_up/880932ee6d8c2bd3/run_manifest.json).
- [Lexical gallery manifest](../outputs/push_up/880932ee6d8c2bd3/lexical_gallery_manifest.json), [semantic gallery manifest](../outputs/push_up/880932ee6d8c2bd3/semantic_gallery_manifest.json), and [overlap gallery manifest](../outputs/push_up/880932ee6d8c2bd3/overlap_gallery_manifest.json).
- [Recorded inventory, provenance and output fingerprints](push_up_run.json).

The galleries contain 86, 50 and 28 video entries respectively, using the original dataset previews. Each gallery initially groups mirrors, displaying 43, 25 and 14 representatives. Video metadata is recorded separately for each gallery. The overlap view shows the representative lexical caption and semantic caption, which can refer to different intervals of the same motion.

Mirror-adjusted overlap is calculated independently of exact-ID overlap. A base ID retrieved as the original in one method and only as its mirror in the other is included in the shared base-ID groups without adding an exact-ID gallery entry.

An additional complete render of motion `000384` from its joints is retained at `outputs/render_review/000384.mp4`: H.264/yuv420p, 20 fps, 199 frames. A sampled frame was inspected previously. The galleries were generated and video metadata inspected; browser interactions were not exercised in this session.

Outputs and caches are ignored by Git. Regenerate them using the command above. The checked-in JSON record preserves this run's counts and fingerprints.

## Provenance and retained findings

The model commit is `e8c3b32edf5434bc2275fc9bab85f82640a19130`; its file signature is `aac4a4b3d9744eaf5eb2588546fbd622448a2f56a7a8b8a49bcd98a474acf288`. The exact run, index and embedding signatures are recorded in the linked manifests and checked-in run record. Native builds and checksums are pinned in `conda-linux-64.lock.txt`, and Python versions in `requirements-linux-cpu.lock.txt`. The existing editable environment runs the updated application without a dependency change.

The four excluded caption-only IDs are `009707`, `011059`, `M009707` and `M011059`. Four paired arrays have squeezed singleton time dimensions. NaN time tags, 1,010 intervals requiring end clipping and eight invalid intervals remain explicitly recorded in the index.

An initial broad semantic query retrieved ordinary arm raises. The final configuration names push-ups or press-ups in every query. This qualitative refinement makes the leading results exercise-specific; it is not a measured precision/recall improvement. Historical searches retain their original query settings in their manifests.
