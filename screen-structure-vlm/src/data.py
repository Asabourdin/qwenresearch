import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent)) #pour éviter les problèmes de path
import config  
from datasets import load_dataset
import random

def _platform_bucket(data_source: str) -> str:
    # screenspot has diff "data_source": iOS, Android, macOS, Windows, gitlab/shop/forum/tool
    # we regroup in 3 categories: mobile, desktop and web

    s = (data_source or "").lower()
    if any(k in s for k in ("ios", "android", "iphone", "mobile")):
        return "mobile"
    if any(k in s for k in ("macos", "windows", "desktop", "mac", "win")):
        return "desktop"
    return "web"  # for the last categories, but also fallback if there are unmapped categories (not the case for screenspot)


def _to_pixel_xyxy(bbox, w: int, h: int, fmt: str = config.SCREENSPOT_BBOX_FORMAT):
    # convert the bbox to absolut pixels ((x1,y1,x2,y2)) depending on the dataset's conventions 
    x0, y0, a, b = [float(v) for v in bbox]

    if max(x0, y0, a, b) <= 1.0:      # if normalised then its probably pixels
        x0, a = x0 * w, a * w         # 0,2 are x; scale by width
        y0, b = y0 * h, b * h         # 1,3 are y; scale by height

    if fmt == "xywh":
        x1, y1, x2, y2 = x0, y0, x0 + a, y0 + b
    else:                              # here "xyxy"
        x1, y1, x2, y2 = x0, y0, a, b

    # stay within image bounds 
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    return (x1, y1, x2, y2)


def load_subset(verbose: bool = True):
    #download screenspot and return the stratified subset list

    ds = load_dataset(config.SCREENSPOT_HF_ID, split="test") #download + on-disk caching (only need the test bc its big enough)
    if verbose:
        print(f"{len(ds)} raw examples loaded from {config.SCREENSPOT_HF_ID}")
        print(f"columns: {ds.column_names}")


    # we name columns variables including different possible names across datasets
    cols = ds.column_names
    img_col = _first(cols, ["image", "img", "screenshot"])
    instr_col = _first(cols, ["instruction", "instructions", "query"])
    bbox_col = _first(cols, ["bbox", "bounding_box", "box"])
    type_col = _first(cols, ["data_type", "type"])
    src_col = _first(cols, ["data_source", "data_souce", "source", "platform"])

    # we sample the sources proportionally to keep balance between text/icon and mobile/desktop/web 
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in range(len(ds)):
        dtype = str(ds[i][type_col]).lower() if type_col else "text"
        plat = _platform_bucket(str(ds[i][src_col]) if src_col else "")
        groups[(dtype, plat)].append(i)

    # seeded rng for reproductibility and also if we run different models/conditions we can still compare with previous runs
    rng = random.Random(config.RANDOM_SEED)
    n_total = len(ds)
    target = min(config.SUBSET_SIZE, n_total)

    chosen: list[int] = []
    for key, idxs in groups.items():
        rng.shuffle(idxs)
        # proportional allocation: each (type, platform) bucket contributes a share of the target subset size proportional to its share of the full dataset
        take = max(1, round(target * len(idxs) / n_total))  # proportional, >=1 per cell
        chosen.extend(idxs[:take])
    # The per-bucket allocation above can overshoot the target number so we shuffle and cut to the exact number we need
    rng.shuffle(chosen)
    chosen = chosen[:target]

    examples = []
    for new_id, i in enumerate(chosen):
        row = ds[i]
        img = row[img_col].convert("RGB")  # normalise to RGB (some source images are grayscale/RGBA)
        w, h = img.size
        examples.append(
            {
                "id": new_id,
                "image": img,
                "orig_w": w,
                "orig_h": h,
                "instruction": str(row[instr_col]),
                "bbox": _to_pixel_xyxy(row[bbox_col], w, h),
                "data_type": str(row[type_col]).lower() if type_col else "text",
                "platform": _platform_bucket(str(row[src_col]) if src_col else ""),
            }
        )

    if verbose:
        # sanity check pretty print
        dist = defaultdict(int)
        for e in examples:
            dist[(e["data_type"], e["platform"])] += 1
        print(f"built subset of {len(examples)} (seed={config.RANDOM_SEED})")
        for k in sorted(dist):
            print(f"        {k[0]:5s} / {k[1]:8s}: {dist[k]}")
    return examples


def _first(available, candidates):
    #helper to pick the right column name
    
    for c in candidates:
        if c in available:
            return c
    return None


if __name__ == "__main__":
    # quick manual check: prints the subset distribution
    load_subset()
