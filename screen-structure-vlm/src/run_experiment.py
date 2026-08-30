"""
For each example we:
    1. pre-resize the image with smart_resize -> (Wr, Hr), save to a temp file
    2. (structure / filtered only) run OCR on that resized image -> prompt block
    3. build the prompt, call the model, parse the point (resized pixels)
    4. rescale the point back to ORIGINAL pixels, score point-in-box
    5. record latency, ocr time, correctness, type, platform

We loop model-OUTER so each model is loaded once (loading is the slow part),
and write results incrementally so a crash never loses completed work.

Usage:
    python src/run_experiment.py                     # all six cells, full subset
    python src/run_experiment.py --limit 20           # quick dry run, 20 examples
    python src/run_experiment.py --models 3B          # only the 3B model
    python src/run_experiment.py --limit 40 --models 3B  # small, one-model check

"""

import argparse
import csv
import sys
import tempfile
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  
from src import data as data_mod  
from src import ocr as ocr_mod 
from src import prompts as prompts_mod  
from src.parse import parse_point  
from src.evaluate import point_in_box, boxes_overlap  
from src.smart_resize import smart_resize  
from src.model import VLM  
from src.evaluate import summarise


#columns for our results.csv
FIELDS = [
    "model", "condition", "id", "instruction", "data_type", "platform",
    "orig_w", "orig_h", "resized_w", "resized_h",
    "pred_x", "pred_y", "gt_x1", "gt_y1", "gt_x2", "gt_y2",
    "correct", "latency_s", "ocr_s", "raw_output",
    "n_ocr_lines", "n_kept", "target_survived",
]


def _prepare(example, tmpdir):
    #resize all imgs once and store them in a temp dir
    h0, w0 = example["orig_h"], example["orig_w"]

    Hr, Wr = smart_resize(
        h0, w0,
        factor=config.IMAGE_FACTOR,
        min_pixels=config.MIN_PIXELS,
        max_pixels=config.MAX_PIXELS,
        max_ratio=config.MAX_ASPECT_RATIO,
    )

    img_r = example["image"].resize((Wr, Hr))

    path = Path(tmpdir) / f"img_{example['id']}.png"
    img_r.save(path)
    return img_r, str(path), Wr, Hr


def run(limit=None, models=None):
    examples = data_mod.load_subset()
    if limit:
        # check we have the correct nb of examples
        # we do it here so that different runs with the sme rng and same limits have the same images
        examples = examples[:limit]

    #we take both models unless specified in the terminal command (in case we want to check things only with the 3B model (or the 7B))
    model_ids = [config.SMALL_MODEL, config.LARGE_MODEL]
    if models:
        wanted = {m.upper() for m in models}
        model_ids = [m for m in model_ids if config.MODEL_SHORT[m].upper() in wanted]

    out_csv = config.RESULTS_DIR / "results.csv" #create the csv

    mem_rows = []  # for the pretty print in the terminal at the end
    with open(out_csv, "w", newline="") as f, tempfile.TemporaryDirectory() as tmp:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        # pre-resize every image once
        prepared, ocr_cache = {}, {}
        for ex in examples:
            prepared[ex["id"]] = _prepare(ex, tmp)

        for model_id in model_ids:
            short = config.MODEL_SHORT[model_id]
            print(f"\nloading {short} ({model_id})")
            #we load models one by one and keep that model for all runs before going to another one to save the loading time
            vlm = VLM(model_id)

            for condition in config.CONDITIONS:
                print(f"{short} / {condition} : {len(examples)} examples")
                for i, ex in enumerate(examples):
                    img_r, path, Wr, Hr = prepared[ex["id"]]

                    ocr_s = 0.0
                    ocr_block = None
                    # stay none for the plain run but are kept for after
                    n_ocr_lines = n_kept = target_survived = None
                    if condition in ("structure", "filtered"):
                        # OCR is run once for both the conditions as we can reuse it to save time
                        if ex["id"] not in ocr_cache:
                            t = time.perf_counter()
                            els = ocr_mod.extract_elements(img_r)
                            ocr_cache[ex["id"]] = (els, time.perf_counter() - t)
                            #we take the ocr once and check the time for perf check

                        els, ocr_s = ocr_cache[ex["id"]]

                        if condition == "structure":
                            # full ocr block
                            ocr_block = ocr_mod.elements_to_prompt_block(els)
                        else:  
                            # esle = filtered
                            # we keep the 5 most relevant elements
                            filtered_els = ocr_mod.filter_elements(
                                els, ex["instruction"], max_keep=5
                            ) 

                            ocr_block = ocr_mod.elements_to_prompt_block(
                                filtered_els, empty_message="(no matching text found)"
                            )
                            # diagnostic vals to check filter behavious 
                            n_ocr_lines = len(els)          # ocr lines before filtering
                            n_kept = len(filtered_els)       # ocr lines kept
                            ow, oh = ex["orig_w"], ex["orig_h"]

                            # we put back the boxes in our non resized space for comparison
                            target_survived = int(any(
                                boxes_overlap(
                                    (b["box"][0] * ow / Wr, b["box"][1] * oh / Hr,
                                     b["box"][2] * ow / Wr, b["box"][3] * oh / Hr),
                                    ex["bbox"],
                                )
                                for b in filtered_els
                            ))

                    # build the prompt
                    prompt = prompts_mod.build_prompt(
                        ex["instruction"], Wr, Hr, ocr_block=ocr_block
                    )
                    raw, latency = vlm.answer(path, prompt)

                    # now we resize the coordinates from the model's answer
                    pt_resized = parse_point(raw, Wr, Hr)
                    pt_orig = None
                    if pt_resized is not None:
                        pt_orig = (
                            pt_resized[0] * ex["orig_w"] / Wr,
                            pt_resized[1] * ex["orig_h"] / Hr,
                        )
                    correct = point_in_box(pt_orig, ex["bbox"])

                    #we write in the csv
                    writer.writerow({
                        "model": short, "condition": condition, "id": ex["id"],
                        "instruction": ex["instruction"], "data_type": ex["data_type"],
                        "platform": ex["platform"], "orig_w": ex["orig_w"],
                        "orig_h": ex["orig_h"], "resized_w": Wr, "resized_h": Hr,
                        "pred_x": None if pt_orig is None else round(pt_orig[0], 1),
                        "pred_y": None if pt_orig is None else round(pt_orig[1], 1),
                        "gt_x1": round(ex["bbox"][0], 1), "gt_y1": round(ex["bbox"][1], 1),
                        "gt_x2": round(ex["bbox"][2], 1), "gt_y2": round(ex["bbox"][3], 1),
                        "correct": int(correct), "latency_s": round(latency, 3),
                        "ocr_s": round(ocr_s, 3),
                        # raw model output is truncated to 300 chars and only serves for mnual checks  
                        "raw_output": raw.replace("\n", " ")[:300],
                        "n_ocr_lines": n_ocr_lines, "n_kept": n_kept,
                        "target_survived": target_survived,
                    })

                    # flush after every row to prevent losing all results from crashes and having to rerun everything
                    f.flush()

                    # keep in memory to report accuracy directly in the pretty print
                    mem_rows.append({
                        "model": short, "condition": condition,
                        "data_type": ex["data_type"], "platform": ex["platform"],
                        "correct": int(correct), "latency_s": latency, "ocr_s": ocr_s,
                        "pred_point": pt_orig,
                    })

                    # we keep a print to checked the run has not silently crashed
                    if (i + 1) % 10 == 0 or (i + 1) == len(examples):
                        print(f"    ... {short}/{condition}: {i + 1}/{len(examples)} done")

            # when all runs are done on a model we free its space from memory to welcome the other model
            del vlm

    print(f"\n run done ! \nwrote {out_csv}")
    _print_summary(mem_rows)
    return out_csv


def _print_summary(mem_rows):
    #pretty print to check everything without having to draw the graphs each time

    cells = {}
    for r in mem_rows:
        cells.setdefault((r["model"], r["condition"]), []).append(r)
    if not cells:
        return

    print(f"\n{'cell':20s} {'n':>4} {'acc':>7} {'text':>7} {'icon':>7}")
    for (model, condition), rows in cells.items():
        s = summarise(rows)
        print(f"{model + '/' + condition:20s} {s['n']:>4} {s['accuracy']:>7.3f} "
              f"{s['accuracy_text']:>7.3f} {s['accuracy_icon']:>7.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only run N examples")
    ap.add_argument("--models", nargs="+", default=None,
                     help="short model names to run (fe --models 3B)")
    run(**vars(ap.parse_args()))
