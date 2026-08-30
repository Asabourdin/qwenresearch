import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  


class VLM:
    def __init__(self, model_id: str):
        # Both imported lazily  so that importing src.model doesn't require mlx_vlm to be installed or working 
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.model_id = model_id
        # `load()` downloads the model on first use (cached under
        # ~/.cache/huggingface after that) and returns both the model weights
        # and the processor (tokenizer + image preprocessor) needed to turn
        # a (prompt, image) pair into model inputs.
        self.model, self.processor = load(model_id)
        self.config = load_config(model_id)

        # IMPORTANT: pin the processor's pixel budget to ours so its internal smart_resize is a no-op on our pre-resized images (keeps coordinates deterministic)
        # mlx-vlm's processor has its OWN internal image resizing step which we do not want bc we need to check the true bounding boxes' coordinates
        # so we set min_pixels/max_pixels to match config.py's values 
        # then its internal resize step sees an image that's already within bounds and leaves it alone
        ip = getattr(self.processor, "image_processor", None)

        if ip is not None:
            if hasattr(ip, "min_pixels"):
                ip.min_pixels = config.MIN_PIXELS
            if hasattr(ip, "max_pixels"):
                ip.max_pixels = config.MAX_PIXELS

    def answer(self, image_path: str, prompt: str) -> tuple[str, float]:
        """Return (raw_text, latency_seconds) for a single image + prompt."""
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        # Qwen2.5-VL needs a prompt wrapped in a specific format (classic vlm to specify the types of the inputs (text vs img))
        formatted = apply_chat_template(
            self.processor, self.config, prompt, num_images=1
        )

        # clock just before generate for the latency measure
        t0 = time.perf_counter()

        out = generate(
            self.model,
            self.processor,
            formatted,
            image=[image_path],
            max_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            verbose=False,
        )

        latency = time.perf_counter() - t0
        
        text = out if isinstance(out, str) else getattr(out, "text", str(out)) #we make sure that vlm.answer() returns a str and not a weird object we wouldn't be able to handle
        return text, latency
