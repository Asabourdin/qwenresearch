import re
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1) # we want it to compute ONCE and than just return the same cached model 
def _engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def extract_elements(pil_image) -> list[dict]:
    #run OCR on the image and get a  list of {text, box_xyxy}

    # we put the img as a numpt array in RGB for compatibility with RapidOCR
    arr = np.array(pil_image.convert("RGB"))
    result, _ = _engine()(arr)  # result: list of [quad_points, text, score] or None
    elements = []
    if not result:
        # RapidOCR returns None (not an empty list) when it finds no text at all so we normalise it as an empty list
        return elements
    for quad, text, score in result:
        # for now the ocr confidence isnt relevant but maybe in the future could help filter
        if not text or not str(text).strip():
            continue
        # RapidOCR returns the boxes as corner points (for rotated and skewed text) 
        # but we work with bounding boxes
        # so we take the bounding box of the 4 corners (left or top corner to right or bottom to fit the whole zone)
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        elements.append(
            {
                "text": str(text).strip(),
                "box": (
                    float(min(xs)),
                    float(min(ys)),
                    float(max(xs)),
                    float(max(ys)),
                ),
            }
        )
    return elements


def elements_to_prompt_block( elements: list[dict], max_elements: int = 60, empty_message: str = "(no text detected)") -> str:
    # turn the list of {text, box_xyxy} into the text block that goes into the model's prompt

    if not elements:
        
        return empty_message # a var and not an empty str bc we want it to account for either "no text found" or "text found but not relevant so filtered"
   
    # we sort so that the text match how we would read in ENGLISH (top to bot, left to right)
    ordered = sorted(elements, key=lambda e: (round(e["box"][1] / 10), e["box"][0]))
    lines = []

    #we cap so that text heavy OCR doesn't blow out the prompt
    # bc we first experienced OOM crashes
    for e in ordered[:max_elements]:
        x1, y1, x2, y2 = (round(v) for v in e["box"])
        lines.append(f'"{e["text"]}" [{x1},{y1},{x2},{y2}]')
    return "\n".join(lines)

_WORD_RE = re.compile(r"[a-z0-9]+") # tokenizer for filter (for latin alphabet (bc we mainly work with english images))


def filter_elements(elements: list[dict], instruction: str, max_keep: int = 5) -> list[dict]:
    #keep only the elements that look relevant to the instruction for the "filtered" condition

    instr_words = set(_WORD_RE.findall(instruction.lower()))
    if not instr_words:
        # if an instruction has no words, we only keep an empty list
        return []

    scored = []
    for idx, el in enumerate(elements):
        text_lower = el["text"].lower()
        el_words = set(_WORD_RE.findall(text_lower))
        # initial score: how many whole words does this OCR line share with the instruction? 
        # we choose not to count duplicates to not skew the score if an element appears multiple times
        score = len(instr_words & el_words)
        
        # second part of the score: we catch if there is a substring match between text and instruction that isnt an exact match
        if any(len(w) >= 3 and w in text_lower for w in instr_words): # not counted per match to not skew and to cap the importance compared to the exact matches
            score += 0.5
        if score > 0:
            scored.append((score, idx, el))

    if not scored:
        # if scored is empty we just return an empty list (nothing will be passed to the prompt)
        return []

    
    scored.sort(key=lambda t: (-t[0], t[1])) # we sort so that we keep the text with highest score
    top = scored[:max_keep]
    
    top.sort(key=lambda t: (round(t[2]["box"][1] / 10), t[2]["box"][0])) #we put back to english reading order (bc it has been ranked in importance order right before)
    return [el for _, _, el in top]
