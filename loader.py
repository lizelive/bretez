from huggingface_hub import hf_hub_download
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def load_image():
    file = hf_hub_download(repo_id="peasanttide/turgot-david-rumsey", filename="sheet_15.jpg", repo_type="dataset")
    im = Image.open(file)
    return im
