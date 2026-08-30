# IMPORTANT: besides the ocr (which depends on the condition), all the prompts will be built in the EXACT same schema to prevent uncertainty in the runs' comparison


# Qwen2.5-VL has been finetuned to answer in a strict minimal format (fixed-shape JSON object) for grounding tasks
# good for us bc it also keeps the raw text short so we prevent OOM crashes
# note: make sure to downstream not strictly require a json because we are still working with a non-deterministic model (even with temp = 0)
TASK_TEMPLATE = (
    "You are shown a {w}x{h} pixel screenshot of a graphical user interface.\n"
    "Task: locate the single UI element that best matches this instruction:\n"
    '  "{instruction}"\n'
    "{structure_block}"
    "Respond with ONLY the pixel coordinates of the point to click, as JSON:\n"
    '  {{"point": [x, y]}}\n'
    "x is in [0, {w}] and y is in [0, {h}]; the origin (0,0) is the top-left corner. "
    "Do not output anything else."
)

# ONLY when ocr_block is not None, we provide the model with a the ocr help
# it is the structure-block, filled with the ocr text produced (full or filtered)

STRUCTURE_HEADER = (
    "To help you, here is the text detected on the screen by OCR, each with its "
    "bounding box as [x1,y1,x2,y2] pixels:\n"
    "{ocr_block}\n"
    "Use these positions when the target is a piece of text. "
    "For icons without text, rely on the image itself.\n"
)


def build_prompt(instruction: str, w: int, h: int, ocr_block: str | None = None) -> str:
    
    # return the complete prompt str
    # for the 3 conditions: plain puts ocr_block = None, structure puts ocr_block = the full ocr str, and filtered puts ocr_block = instruction relevant ocr str
    if ocr_block is None:
        structure_block = ""
    else:
        structure_block = STRUCTURE_HEADER.format(ocr_block=ocr_block)
    return TASK_TEMPLATE.format(
        w=w, h=h, instruction=instruction, structure_block=structure_block
    )
