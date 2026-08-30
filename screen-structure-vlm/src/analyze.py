import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  
from src.evaluate import summarise  
import csv
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt


def _load_rows():
    #read results and returns it as a list of dicts with the correct types
    
    rows = []
    with open(config.RESULTS_DIR / "results.csv") as f:
        for r in csv.DictReader(f):
            r["correct"] = int(r["correct"])
            r["latency_s"] = float(r["latency_s"])
            r["ocr_s"] = float(r["ocr_s"] or 0.0)
            r["pred_point"] = None if r["pred_x"] in ("", "None") else (r["pred_x"], r["pred_y"]) #if a value is returned then we keep it, but if it returns None (parse failure) or an empty string (nothing parsed) we keep none
            #put filter columns as None for the plain and structure rows
            for k in ("n_ocr_lines", "n_kept", "target_survived"):
                r[k] = None if r.get(k) in (None, "", "None") else int(r[k])
            rows.append(r)
    return rows


def analyze():
    rows = _load_rows()

    # group every row by model and condition
    cells = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["condition"])].append(r)

    # fixed affichage order for the six cells (by model then condition) to protect from new runs (fe: running just the 7B model)
    order = [
        ("3B", "plain"), ("3B", "structure"), ("3B", "filtered"),
        ("7B", "plain"), ("7B", "structure"), ("7B", "filtered"),
    ]
    summ = {c: summarise(cells[c]) for c in order if cells[c]} #to handle a partial csv (fe: running just the 3B)

    # pretty print 
    print(f"\n{'cell':16s} {'n':>4} {'acc':>7} {'text':>7} {'icon':>7} "
          f"{'lat(s)':>7} {'ocr(s)':>7} {'pfail':>6}")
    for c in order:
        if c not in summ:
            continue
        s = summ[c]
        print(f"{c[0]+'/'+c[1]:16s} {s['n']:>4} {s['accuracy']:>7.3f} "
              f"{s['accuracy_text']:>7.3f} {s['accuracy_icon']:>7.3f} "
              f"{s['mean_latency_s']:>7.2f} {s['mean_ocr_s']:>7.2f} "
              f"{s['parse_fail_rate']:>6.2f}")

    # write summary.csv for paper and graphs

    keys = ["model", "condition", "n", "accuracy", "accuracy_text", "accuracy_icon",
            "accuracy_mobile", "accuracy_desktop", "accuracy_web",
            "mean_latency_s", "mean_ocr_s", "parse_fail_rate"]
    with open(config.RESULTS_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for c in order:
            if c in summ:
                w.writerow({"model": c[0], "condition": c[1], **{k: summ[c][k] for k in keys[2:]}})

    _figures(summ, order)
    _filter_survival_analysis(rows)
    print(f"\nsummary.csv written & figures in {config.RESULTS_DIR}")


def _filter_survival_analysis(rows):
    # to check if accuracy is going higher when filtered ocr is kept or when there is none
    
    sub = [r for r in rows
           if r["model"] == "3B" and r["condition"] == "filtered"
           and r["target_survived"] is not None]
    if not sub:
        print("\nno 3B/filtered rows with target_survived data: investigate")
        return

    groups = {0: [], 1: []}
    for r in sub:
        groups[r["target_survived"]].append(r)

    print(f"\n{'target_survived':16s} {'n':>4} {'acc':>7}")
    accs = {}
    for k in (0, 1):
        g = groups[k]
        acc = (sum(r["correct"] for r in g) / len(g)) if g else float("nan")
        accs[k] = acc
        print(f"{k:<16d} {len(g):>4} {acc:>7.3f}")

    labels = [f"survived=0\n(n={len(groups[0])})", f"survived=1\n(n={len(groups[1])})"]
    vals = [accs[0], accs[1]]
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ax.bar(labels, vals, color=["saddlebrown","darkgreen"])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("3B/filtered: accuracy depending on whether the ocr survived filtering")
    for i, v in enumerate(vals):
        if v == v:  # skip NaN for no syntax errors !!
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "fig_filter_survival.png", dpi=150)


def _figures(summ, order):
    #general accuracy graph
    labels = [f"{m}\n{c}" for (m, c) in order if (m, c) in summ]
    accs = [summ[c]["accuracy"] for c in order if c in summ]

    #pretty palette jeje
    palette = {
        ("3B", "plain"): "#a1d99b", ("3B", "structure"): "#41ab5d", ("3B", "filtered"): "#006d2c",
        ("7B", "plain"): "#d9b382", ("7B", "structure"): "#a9682f", ("7B", "filtered"): "#5c3a21",
    }
    colors = [palette[c] for c in order if c in summ]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.bar(labels, accs, color=colors)
    ax.set_ylabel("ScreenSpot accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Grounding accuracy")
    for i, v in enumerate(accs):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "fig_accuracy.png", dpi=150)

    # text vs icon per cell 
    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
    cells = [c for c in order if c in summ]
    import numpy as np
    x = np.arange(len(cells))
    tw = [summ[c]["accuracy_text"] for c in cells]
    ic = [summ[c]["accuracy_icon"] for c in cells]
    ax2.bar(x - 0.2, tw, 0.4, label="text", color="saddlebrown")
    ax2.bar(x + 0.2, ic, 0.4, label="icon", color="darkgreen")
    ax2.set_xticks(x, [f"{m}/{c}" for (m, c) in cells], fontsize=8)
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0, 1)
    ax2.set_title("Text vs icon")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(config.RESULTS_DIR / "fig_by_type.png", dpi=150)


if __name__ == "__main__":
    analyze()
