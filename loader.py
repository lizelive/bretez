from huggingface_hub import hf_hub_download
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def load_image():
    file = hf_hub_download(repo_id="peasanttide/turgot-david-rumsey", filename="sheet_15.jpg", repo_type="dataset")
    im = Image.open(file)
    return im

TILE_SIZE = 512

x = 5050
y = 5050

load_image().crop((x, y, x + TILE_SIZE, y + TILE_SIZE)).save("cropped.png")