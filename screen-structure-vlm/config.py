from pathlib import Path

# paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"          
RESULTS_DIR = ROOT / "results"    
DATA_DIR.mkdir(exist_ok=True) #check and create if needed the folders to put our results and data
RESULTS_DIR.mkdir(exist_ok=True)

# Models 
# two sizes of the SAME family so only change is scale

# model IDs = HF repo names
SMALL_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
LARGE_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

# model name for prints, results graphs and csvs
MODEL_SHORT = {
    SMALL_MODEL: "3B",
    LARGE_MODEL: "7B",
}

# Prompt structure
CONDITIONS = ["plain", "structure", "filtered"]

 # plain screenshot vs screenshot + full ocr vs screenshot + short ocr if relevant 

# HF dataset id for ScreenSpot
SCREENSPOT_HF_ID = "rootsautomation/ScreenSpot" #ootsautomation stores bbox as [x1, y1, x2, y2] so we normalise them to absolute pixels for qwen

# bbox layout
SCREENSPOT_BBOX_FORMAT = "xyxy"      #  rootsautomation stores corner coordinates (x1,y1,x2,y2) but before I had x/y/width/height is ambiguous per-box so we declare it here and apply it to prevent format issues (which would do weird results but wouldn't crash necessarily so we wouldn't see it directly)

# Number of examples to evaluate
# full set is more than 1200 but bc we're working on a local laptop, I cut it to keep total runtime acceptable
# Important to have a blocked number and a fixed seed so that the same subset is used for ALL four conditions.
SUBSET_SIZE = 240
STRATIFY_KEYS = ("data_type", "platform")   # for text/icon and mobile/desktop/web balance (bc they are arranged in the dataset so if we only pick the first 240 we'll miss web images)
RANDOM_SEED = 42                            

# Resizing for Qwen 

# Main issue: qwen2.5 first resize the image and then emits coordinates of icons and text in that new space. 
# BUT we want to feed it the coordinates we have found
# SO we need to resize the images through qwen's resizing algo, feed it to the model, read the resized coordinates in that space and rescale back to the original image to compare against the ground truth box

IMAGE_FACTOR = 28                    # patch(14) * merge(2); smart_resize rounds to this
MIN_PIXELS = 56 * 56                 # lower bound Qwen uses
MAX_PIXELS = 1280 * 28 * 28          # to keep big images manageable
MAX_ASPECT_RATIO = 200               # from the official implementation

# vlm params
MAX_NEW_TOKENS = 128                 # we technically need very little and want to keep fast iterations
TEMPERATURE = 0.0                    # bc we want deterministic decoding 

# coordinates returned management
COORD_INTERPRETATION = "pixel"       # checked agaist the smoke test imgs: we pick "pixels" bc it locates the correct element (+ makes sense bc qwen emits absolute pixels of the resized img)
