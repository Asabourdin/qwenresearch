import math


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 1280 * 28 * 28,
    max_ratio: int = 200,
) -> tuple[int, int]:
    # Qwen2.5-VL returns coordinates in the pixel space of the image after having an internal resize
    # but we need to compare the returned coordinates with the ground truth bounding boxes so we need to resize the images BEFORE feeding them to qwen so we know exactly what point in qwens answer maps to what point in the original img 
    # so we put this fct to resize the images following its internal resizing algo

    # we reimplement the function in qwen_vl_utils
    # this means we round each side of the img to a multiple of factor (28 = patch (14) x merge (2)) 
    # then we keep the total pixel count within [min_pixels, max_pixels] and the aspect ratio within the accepted bound.

    # first we refuse image shapes so extreme that "resize while keeping the aspect ratio" wouldn't make sense 
    # qwen does it naturally without error or warning msg so we do it too to work on the same inputs 
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"absolute aspect ratio must be < {max_ratio}, "
            f"got {max(height, width) / min(height, width):.1f}"
        )

    #round each side to the nearest multiple of `factor` (28), with a floor of one full factor so we never round to zero

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)

    if h_bar * w_bar > max_pixels:
        # if we have more pixels than the maximum accepted by the model, we shrink the image by a factor beta so that it reaches the max nb of pixels
        beta = math.sqrt((height * width) / max_pixels)

        # we shrink both sides so the aspect ration is preserved
        h_bar = max(factor, math.floor(height / beta / factor) * factor) # and we re-round to a multiple of factor
        w_bar = max(factor, math.floor(width / beta / factor) * factor)

    elif h_bar * w_bar < min_pixels:
        
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor #ceil and not floor bc we want to have "at least enough pixels"
        w_bar = math.ceil(width * beta / factor) * factor

    # else: the img is already landed between min_pixels and max_pixels

    return h_bar, w_bar
