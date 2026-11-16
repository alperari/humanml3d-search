# First local push-up run

Completed on 2026-11-16 in the isolated `humanml3d-search` conda environment, using the original dataset at `/home/aari/data_share/HumanML3D`. The Fit3D conversion project and its environment were not modified. This record describes real-data execution; no automated tests were added or run.

## Completed workflow

```bash
conda activate humanml3d-search
humanml3d-search index --config configs/local.yml
humanml3d-search model --config configs/local.yml
humanml3d-search run --config configs/local.yml --activity push_up --offline
```

The dataset snapshot contains 29,232 caption files and 87,384 caption rows. Eligibility retains 29,228 motion IDs and 87,372 caption rows, or 14,614 base IDs. Encoding the 55,858 unique eligible captions took approximately 398 seconds on CPU with eight threads and batch size 32. The final command reused those embeddings and completed offline.

| Selection | Distinct motion IDs | Base IDs after grouping mirrors |
| --- | ---: | ---: |
| Lexical | 86 | 43 |
| Semantic | 50 | 25 |
| Both methods | 28 | 14 |
| Lexical only | 58 | 29 |
| Semantic only | 22 | 11 |
| Union | 108 | 54 |

The lexical representatives comprise 56 explicit-name matches and 30 contextual matches. All 50 semantic CSV rows have distinct motion IDs and ranks 1–50. Their cosine scores range from approximately 0.8580 to 0.7913. Motion `009006` and its mirrored counterpart lead the ranking with the original caption “the person does multiple push-ups”. These are search candidates; relevance has not been manually assigned to all retrieved motions.

## Result artifacts

- [Lexical CSV](../outputs/push_up/e7019ad8989c9098/lexical_results.csv)
- [Semantic CSV](../outputs/push_up/e7019ad8989c9098/semantic_results.csv)
- [Combined motion records and counts](../outputs/push_up/e7019ad8989c9098/combined_results.json)
- [Caption evidence](../outputs/push_up/e7019ad8989c9098/caption_matches.jsonl)
- [Video review gallery](../outputs/push_up/e7019ad8989c9098/review.html)
- [Search manifest](../outputs/push_up/e7019ad8989c9098/run_manifest.json)
- [Gallery manifest](../outputs/push_up/e7019ad8989c9098/gallery_manifest.json)
- [Recorded inventory and output fingerprints](push_up_run.json)

All 108 gallery videos reuse original dataset previews. FFprobe reports H.264, yuv420p and 20 fps for every video; their frame counts match the indexed joint sequences. The default gallery view shows 54 representatives. An additional complete render of retrieved motion `000384` from its joints is saved at `outputs/render_review/000384.mp4`, with a render manifest: H.264/yuv420p, 20 fps, 199 frames. A sampled frame was inspected for the skeleton view. The interactive gallery was generated and its media inspected; browser interactions were not exercised in this session.

Outputs and caches are ignored by Git. Regenerate them with the commands above when cloning the source elsewhere. The checked-in JSON record preserves this run's counts and fingerprints independently of those working directories.

## Provenance

- Model commit: `e8c3b32edf5434bc2275fc9bab85f82640a19130`.
- Model-file signature: `aac4a4b3d9744eaf5eb2588546fbd622448a2f56a7a8b8a49bcd98a474acf288`.
- Index signature: `802cb15816d05275bdeda3e33f87699395b7ab28bf70fc8ead5b6868c7cf8888`.
- Embedding signature: `da13c1eba518fda18eee645b0028fa7aab47c44f61f036d6a92e1b021fa919fb`.
- Final search signature: `e7019ad8989c9098d751b1ddbd902645bbe938ee972c552b6deb3c00a6723da3`.

Native package builds and checksums are pinned in `conda-linux-64.lock.txt`; Python package versions are pinned in `requirements-linux-cpu.lock.txt`. `python -m pip check` reported no broken requirements in the activated environment.

## Findings retained

The four excluded caption-only IDs are `009707`, `011059`, `M009707` and `M011059`. Four paired arrays have squeezed one-frame dimensions. The original data also contains NaN time tags, 1,010 caption intervals requiring end clipping and eight invalid intervals. The index retains the original tags/shapes and reports these cases explicitly.

An initial semantic query described lowering and raising the body using the arms without naming the exercise. With maximum-over-queries scoring, that query strongly retrieved ordinary arm raises; its leading caption scored 0.8733. The final activity configuration uses push-up/press-up names in every query. The leading candidates now describe push-ups, and exact-ID lexical/semantic overlap rose from six to 28. This is a qualitative query refinement, not a measured precision/recall improvement. The initial run remains under `outputs/push_up/6a20c987ef00598f/` with its original query settings in its manifest.
