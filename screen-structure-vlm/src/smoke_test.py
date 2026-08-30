# mainly for coordinates mapping to understand qwen's resize and usual coordinates naming
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  
from src import data as data_mod  
from src import prompts as prompts_mod  
from src.parse import parse_point  
from src.evaluate import point_in_box  
from src.smart_resize import smart_resize  
from src.model import VLM  


def run(n=10):
    from PIL import ImageDraw #lazy import bc not always used

    out_dir = config.RESULTS_DIR / "smoke"
    out_dir.mkdir(exist_ok=True)

    # only plain and small model bc coordinates parsing is similar for both models and we just want the fastest run and don't need the ocr
    examples = data_mod.load_subset()[:n]
    vlm = VLM(config.SMALL_MODEL)
    hits = 0

    with tempfile.TemporaryDirectory() as tmp:
        for ex in examples:
            # same smart_resize + prompt + answer + parse + rescale loop than in the normal run
            # skip ocr
            h0, w0 = ex["orig_h"], ex["orig_w"]
            Hr, Wr = smart_resize(
                h0, w0, config.IMAGE_FACTOR,
                config.MIN_PIXELS, config.MAX_PIXELS, config.MAX_ASPECT_RATIO,
            )
            img_r = ex["image"].resize((Wr, Hr))
            path = Path(tmp) / f"s_{ex['id']}.png"
            img_r.save(path)

            prompt = prompts_mod.build_prompt(ex["instruction"], Wr, Hr, ocr_block=None)
            raw, latency = vlm.answer(str(path), prompt)
            pt_r = parse_point(raw, Wr, Hr)
            pt_o = None
            if pt_r is not None:
                pt_o = (pt_r[0] * w0 / Wr, pt_r[1] * h0 / Hr)
            correct = point_in_box(pt_o, ex["bbox"])
            hits += int(correct)

            # draw a copy of the image comparing the ground truth and the model guess
            canvas = ex["image"].copy()
            d = ImageDraw.Draw(canvas)
            d.rectangle(ex["bbox"], outline=(0, 200, 0), width=3)  # green box = ground truth
            if pt_o is not None:
                x, y = pt_o
                d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(220, 0, 0))  # red point = model's guess
            canvas.save(out_dir / f"{ex['id']:03d}_{'OK' if correct else 'MISS'}.png")

            # also print the raw model output to check if its the model's answer that was wrong or if its the coordinates' handling that is wrong
            print(f"[{ex['id']:03d}] {ex['data_type']:5s} {ex['platform']:8s} "
                  f"{'OK ' if correct else 'MISS'}  {latency:5.1f}s  "
                  f"raw={raw.strip()[:60]!r}")

    print(f"\nsmoke accuracy: {hits}/{len(examples)}  "
          f"(in {out_dir})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    run(**vars(ap.parse_args()))
