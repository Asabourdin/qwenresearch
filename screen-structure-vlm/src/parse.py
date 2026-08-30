import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# number regex form that matches any integer or decimal number
_NUM = re.compile(r"-?\d+\.?\d*")

# bracket regex that matches only a bracket group whose content is a pure numeric list (to find coordinates)
# bc first mistake in parsing the json wrapper: it would take the first bracket containing a number (even if in the key name )

_BRACKET = re.compile(r"[\[\(]\s*(-?\d+\.?\d*(?:\s*,\s*-?\d+\.?\d*)+)\s*[\]\)]")


def _extract_numbers(text: str) -> list[float]:
    m = _BRACKET.search(text)
    if m:
        # if there is a clean bracket group we just take the numbers out of it
        src = m.group(1)
    else:
        # fallback if no brackets we scan for just numbers, but remove words we know can cause wrong parsing (fe in the keys (point_2d))
        src = re.sub(r"(?i)_?2d\b|point|bbox|box", " ", text)
    return [float(n) for n in _NUM.findall(src)]


def parse_point(text: str, resized_w: int, resized_h: int):
    # we parse the numbers into (x,y) in the resized space (and none if nothing parseable)
    if not text:
        #no text = no coordinates
        return None
    #if text we extract the numbers out of it
    nums = _extract_numbers(text)
    if len(nums) < 2:
        # if less then 2 numbers parsed: its not coordinates, we can't do anything with it
        return None

    # Prefer an explicit 4-number bbox if present (take its centre); else the
    # first two numbers as a point.
    #
    # Why check for 4 numbers AND a "bbox"/"box" keyword, rather than just "4
    # numbers means a bbox"? Because a point answer could legitimately be
    # followed by other unrelated numbers later in the text (e.g. a
    # confidence score, or numbers inside a "label" string) -- 4 numbers
    # existing somewhere in the answer doesn't necessarily mean the model
    # meant to describe a box. Requiring the word "bbox" or "box" to appear
    # too makes this fallback path only trigger when the model has actually
    # signalled "this is a box" (e.g. answered with "bbox_2d" instead of
    # "point_2d" or "point").
    if len(nums) >= 4 and _looks_like_bbox(text):
        # if we have 4 numbers and the text has "bbox" or "box" keyword we highly probably have the coordinates of a box
        # so we take (x,y) as the center of the box
        x1, y1, x2, y2 = nums[:4]
        x, y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # if we don't do that we would have a corner of the box as a point which would increase the risk of mistakes + not respect the vlm's answer (saw in smoke tests)
    else:
        x, y = nums[0], nums[1]

    x, y = _to_resized_pixels(x, y, resized_w, resized_h) # convert into pixels of the resized img for comparison
    # Clamp so near misses still get scored as passes
    x = max(0.0, min(x, resized_w)) 
    y = max(0.0, min(y, resized_h))
    return (x, y)


def _looks_like_bbox(text: str) -> bool:
    return "bbox" in text.lower() or "box" in text.lower()


def _to_resized_pixels(x, y, w, h):
    # this basically check the case where the model ignored the instructions and returned too small fractions of the image instead of pixels' coordinates
    # so that we can still treat them as pixels afterwards
    mode = config.COORD_INTERPRETATION

    #rescale depending on set mode
    if mode == "norm1":            # fractions 0..1 of the image
        return x * w, y * h
    if mode == "norm1000":         # 0..1000 grid
        return x / 1000.0 * w, y / 1000.0 * h
    
    # if both pixels are too tiny to be pixels (if their abs are both smaller to 1.5) we consider that it is probable that the coordinates were not actual pixels and we still scale up 
    # bc too low to make sense in the imgs
    if max(abs(x), abs(y)) <= 1.5:
        return x * w, y * h
    return x, y
