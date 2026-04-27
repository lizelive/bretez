from huggingface_hub import hf_hub_download
from PIL import Image

from .config import INPUT_IMAGE_DOWNSCALE_FACTOR

Image.MAX_IMAGE_PIXELS = None


def load_image(downscale_factor: int = INPUT_IMAGE_DOWNSCALE_FACTOR) -> Image.Image:
    file = hf_hub_download(repo_id="peasanttide/turgot-david-rumsey", filename="sheet_15.jpg", repo_type="dataset")
    im = Image.open(file)
    return downscale_image(im, downscale_factor)


def downscale_image(image: Image.Image, factor: int) -> Image.Image:
    if factor < 1:
        raise ValueError(f"Downscale factor must be at least 1, got {factor}.")
    if factor == 1:
        return image

    width, height = image.size
    scaled_size = (max(1, width // factor), max(1, height // factor))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    scaled_image = image.resize(scaled_size, resampling)
    filename = getattr(image, "filename", None)
    if filename:
        scaled_image.filename = filename
    return scaled_image
