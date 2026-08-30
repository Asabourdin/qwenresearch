from collections import defaultdict

# we score on 'click accuracy' (= a prediction is correct iff the predicted point falls inside the ground-truth bounding box)

# we compute: 
# overall accuracy
# accuracy by data_type (text / icon)
# accuracy by platform (mobile / desktop / web)
# mean latency (and mean OCR latency for structure)
# parse-failure rate (predictions we could not read)

def point_in_box(point, box) -> bool:
    
    # we take point and box in the SAME COORDINATES SPACE
    if point is None:
        # a prediction we couldn't parse a point out of is scored as wrong
        return False
    x, y = point 
    x1, y1, x2, y2 = box
    return (x1 <= x <= x2) and (y1 <= y <= y2) # <= and not < so a point that lands EXACTLY on the box's edge still counts as a hit 


def boxes_overlap(box_a, box_b) -> bool:
    # IoU > 0
    # mainly for debug
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2 # here < and not <= bc we want box to actually overlapp and not just have touching edges


def summarise(rows: list[dict]) -> dict:
    # we want a list of per img result dicts for future summary dict
    def acc(subset):
        # we return the accuracy ratio if there is sth in the subset and nan if there isnt
        return (sum(r["correct"] for r in subset) / len(subset)) if subset else float("nan")

    # we already separate by condition and platform
    by_type = defaultdict(list)
    by_plat = defaultdict(list)
    for r in rows:
        by_type[r["data_type"]].append(r)
        by_plat[r["platform"]].append(r)

    n = len(rows)
    return {
        "n": n,
        "accuracy": acc(rows),
        "accuracy_text": acc(by_type.get("text", [])),
        "accuracy_icon": acc(by_type.get("icon", [])),
        "accuracy_mobile": acc(by_plat.get("mobile", [])),
        "accuracy_desktop": acc(by_plat.get("desktop", [])),
        "accuracy_web": acc(by_plat.get("web", [])),
        "mean_latency_s": (sum(r["latency_s"] for r in rows) / n) if n else float("nan"),
        # for the plain condition ocr_s is 0 (bc no OCR ran)
        "mean_ocr_s": (sum(r.get("ocr_s", 0.0) for r in rows) / n) if n else 0.0,
        # how often did we fail parsing the vlm's answer
        "parse_fail_rate": (sum(1 for r in rows if r["pred_point"] is None) / n) if n else float("nan"),
    }
