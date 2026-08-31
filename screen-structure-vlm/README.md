# Can explicit screen structure substitute for VLM scale?

A small, reproducible study testing whether augmenting a vision-language model
with an explicit structural signal (OCR text + positions) lets a **small**
model close the gap to a **larger** model of the same family on GUI
grounding — i.e. can structure substitute for scale?

> **Research question:** Can giving a small vision-language model (Qwen2.5-VL-3B) explicit structural informationz (OCR text + bounding boxes) let it match a larger model of the same family (Qwen2.5-VL-7B) on GUI grounding?

- **Benchmark:** [ScreenSpot](https://huggingface.co/datasets/rootsautomation/ScreenSpot) (public; GUI grounding, ~1,272 instructions, text/icon elements across mobile/desktop/web screenshots).
- **Models:** Qwen2.5-VL-3B and Qwen2.5-VL-7B, 4-bit quantized, running
  entirely locally via `mlx-vlm` — no API calls, no per-token cost, fully reproducible given the same weights and the same seeded example subset.
- **Design:** a 2×3 grid — {3B, 7B} × {plain screenshot, screenshot + full OCR
  "structure", screenshot + instruction-filtered OCR "filtered"}. Six cells
  in total. (The "filtered" condition was added mid-project)
- **Metric:** click accuracy — a prediction counts as correct iff the
  predicted point falls inside the ground-truth bounding box — reported
  overall and split by element type (text/icon) and platform
  (mobile/desktop/web), plus latency.

---

## What we found

Full run, 240 examples per cell, all six cells:

```
cell                n     acc    text    icon  lat(s)  ocr(s)  pfail
3B/plain          240   0.829   0.902   0.741    8.51    0.00   0.00
3B/structure      240   0.296   0.220   0.389   11.68    0.76   0.00
3B/filtered       240   0.675   0.720   0.620    9.59    0.76   0.00
7B/plain          240   0.846   0.909   0.769   15.47    0.00   0.00
7B/structure      240   0.858   0.947   0.750   20.30    0.76   0.00
7B/filtered       240   0.867   0.955   0.759   15.93    0.76   0.00
```

1. **Structure did not help the small model: it worsened it**
   3B/structure (0.296) is  worse than 3B/plain (0.829), especially on text targets (0.902 → 0.220). This is the opposite of the original hypothesis. Investigating why (through a diagnosis script) showed it wasn't a bug in the ocr (the injected OCR list usually DID contain an accurate correctly-positioned line for the true target), but the 3B model would still pick a different, wrong element once a list of many OCR candidates was in front of it. 
   It looks like the small model was distracted by a long list of mostly irrelevant structural details.

2. **Filtering the OCR list down to maximum 5 instruction-relevant lines recovers some, but not all, of that loss.** 
   3B/filtered (0.675) sits between 3B/plain and 3B/structure, well above the collapse, but still below the plain baseline. 
   The text/icon split shows the same pattern (text: 0.220 → 0.720, still short of plain's 0.902).

3. **The gap to the 7B model is not closed.** 
   3B+structure (0.296) is far below 7B+plain (0.846); even 3B+filtered (0.675) doesn't reach it. The original hypothesis — "structure substitutes for scale" — is not supported for this model pair, at least not for the *full* structure condition. Even if it is better than the full structure, even the filtered one doesn't provide the expected progress.

4. **The 7B model benefits slightly from structure.** 
   This is a very interesting detected trend that would need future investigation.
   We see 7B/plain (0.846) → 7B/structure (0.858) → 7B/filtered (0.867): a small monotonic improvement. 
   We can suppose that a model large enough to not get distracted by a long OCR list could make some use of it. This would mean that the *harm* from structure in this study is specific to the smaller model, not a property of adding OCR text in general.

---

## Repo layout

```
config.py                 ALL knobs: models, subset size, resize budget conditions, coord mode
src/
  smart_resize.py          Qwen's own image-resize algorithm (pre-resize to known coord space)
  data.py                  load ScreenSpot, build the stratified/seeded subset, normalise bboxes
  ocr.py                   RapidOCR wrapper: extract text elements, render prompt block, filter
  prompts.py               builds the model prompt (identical across conditions except OCR block)
  model.py                 thin wrapper around mlx-vlm (load once, .answer(image, prompt))
  parse.py                 model's raw text -> click point in resized-image pixels (robust parser)
  evaluate.py               scoring: point-in-box, box-overlap, per-cell accuracy summaries
  run_experiment.py         the 2x3 loop -> results/results.csv  
  smoke_test.py             checks for the coordinates handling by manual look
  analyze.py                results.csv -> summary.csv + figures + headline prints
  diagnose_structure.py     read-only diagnostic: why did 3B/structure fail? (side-by-side raw output)
results/                    everything the pipeline produces (CSVs, overlays, figures)
data/                       ScreenSpot cache 
```


---

## Setup

Requirements: Python 3.11, around 10 GB free disk (model weights + dataset cache) (done initially with a 16 GB laptop memory)

```bash
cd screen-structure-vlm
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Reproduce

Run in this order — steps 1–2 are gates, not optional:

```bash
python src/data.py                    # 1. sanity-check the dataset loads correctly
python src/run_experiment.py          # 2. the full run (240 examples)
python src/analyze.py                 # 3. run the analysis for numbers and graphs
```

The subset is fixed by `RANDOM_SEED` in `config.py`, decoding is greedy
(`TEMPERATURE=0.0`), and every cell shares the identical resize/coordinate
pipeline, so results are deterministic given the same model weights and are
directly comparable across cells.

## Note on assistance

Parts of this code were debugged using claude code, and the readme was written by it.
Everything was proofread and tested by a human user (me).

