import argparse
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  
from src import data as data_mod  
from src import ocr as ocr_mod  
from src import prompts as prompts_mod  
from src.parse import parse_point  
from src.evaluate import point_in_box  
from src.smart_resize import smart_resize  
from src.model import VLM  
#we saw that we had bad results for text inputs in the vlm (find "cancel")
#so we want to check if there is a mistake in the input, output, parsing or if the model is just worsen by the OCR input

def run(limit=8):
    # we take a slice of limit text examples that we will check in details 
    examples = data_mod.load_subset()[:limit] 
    text_examples = [e for e in examples if e["data_type"] == "text"]
    print(f"we check {len(text_examples)} text examples out of {len(examples)} "
          f"in the first {limit}\n")

    vlm = VLM(config.SMALL_MODEL) #initiate vlm

    with tempfile.TemporaryDirectory() as tmp:
        for ex in text_examples:
            # resize step !
            h0, w0 = ex["orig_h"], ex["orig_w"]
            Hr, Wr = smart_resize(
                h0, w0, config.IMAGE_FACTOR,
                config.MIN_PIXELS, config.MAX_PIXELS, config.MAX_ASPECT_RATIO,
            )
            img_r = ex["image"].resize((Wr, Hr))
            path = Path(tmp) / f"img_{ex['id']}.png"
            img_r.save(path)

            print("=" * 100)
            print(f"[{ex['id']:03d}] instruction: {ex['instruction']!r}")
            print(f"      original: {w0}x{h0}   resized: {Wr}x{Hr}")
            print(f"      GT bbox (orig px): {tuple(round(v, 1) for v in ex['bbox'])}")

            # we check the plain input and output
            prompt_plain = prompts_mod.build_prompt(ex["instruction"], Wr, Hr, ocr_block=None)
            raw_plain, lat_plain = vlm.answer(str(path), prompt_plain)
            pt_r_plain = parse_point(raw_plain, Wr, Hr)
            pt_o_plain = (
                (pt_r_plain[0] * w0 / Wr, pt_r_plain[1] * h0 / Hr)
                if pt_r_plain is not None else None
            )
            ok_plain = point_in_box(pt_o_plain, ex["bbox"])

            print(f"\n Plain:({lat_plain:.1f}s)")
            print(f"    raw: {raw_plain.strip()}")
            print(f"    parsed point (resized px): {pt_r_plain}")
            print(f"    parsed point (orig px):    "
                  f"{tuple(round(v, 1) for v in pt_o_plain) if pt_o_plain else None}"
                  f"    {'OK' if ok_plain else 'MISS'}")

            # we check the same for structured run
            els = ocr_mod.extract_elements(img_r)
            ocr_block = ocr_mod.elements_to_prompt_block(els)
            prompt_struct = prompts_mod.build_prompt(ex["instruction"], Wr, Hr, ocr_block=ocr_block)
            raw_struct, lat_struct = vlm.answer(str(path), prompt_struct)
            pt_r_struct = parse_point(raw_struct, Wr, Hr)
            pt_o_struct = (
                (pt_r_struct[0] * w0 / Wr, pt_r_struct[1] * h0 / Hr)
                if pt_r_struct is not None else None
            )
            ok_struct = point_in_box(pt_o_struct, ex["bbox"])

            print(f"\n Structure ({lat_struct:.1f}s, {len(els)} OCR elements)")
            print("     injected OCR block:")
            for line in ocr_block.splitlines():
                print(f"    {line}")
            print(f"    raw: {raw_struct.strip()}")
            print(f"    parsed point (resized px): {pt_r_struct}")
            print(f"    parsed point (orig px):    "
                  f"{tuple(round(v, 1) for v in pt_o_struct) if pt_o_struct else None}"
                  f"    {'OK' if ok_struct else 'MISS'}")
            print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    run(**vars(ap.parse_args()))
