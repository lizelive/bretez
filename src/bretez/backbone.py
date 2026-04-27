from transformers import AutoImageProcessor, AutoModel

import torch

SAT_MODEL_BIG = "facebook/dinov3-vit7b16-pretrain-sat493m"
SAT_MODEL_SMALL = "facebook/dinov3-vitl16-pretrain-sat493m"
from PIL import Image

import itertools

import logging
logger = logging.getLogger(__name__)

class Backbone:
    def __init__(self, model_name=SAT_MODEL_SMALL):
        self.model = AutoModel.from_pretrained(model_name).to("cuda").eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    def __call__(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to("cuda")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def process_big_image(self, image: Image.Image) -> torch.Tensor: 
        "splits the image into overlappying patches and processes them, then averages the features"
        logger.info(f"Processing big image of size {image.size}")
        tile_size = self.processor.size.width
        stride = tile_size // 2
        width, height = image.size
        width_patches = (width - tile_size) // stride + 1
        height_patches = (height - tile_size) // stride + 1
        patch_size = self.model.config.patch_size
        patch_output_size = tile_size // patch_size
        stride_output_size = stride // patch_size
        num_reg = getattr(self.model.config, "num_register_tokens", 0)
        with torch.inference_mode():
            out_features = torch.zeros(
                (
                    (width_patches - 1) * stride_output_size + patch_output_size,
                    (height_patches - 1) * stride_output_size + patch_output_size,
                    self.model.config.hidden_size,
                )
            )
            feature_counts = torch.zeros(out_features.shape[:2], dtype=out_features.dtype)
            logger.debug(f"Processing image of size {image.size} with patch size {tile_size} and stride {stride}")
            patch_cords = list(
                itertools.product(
                    range(0, width - tile_size + 1, stride),
                    range(0, height - tile_size + 1, stride),
                )
            )
            # batch and process patches
            batch_size = 16

            assert len(patch_cords) % batch_size == 0, "Number of patches must be divisible by batch size"
            
            for i in range(0, len(patch_cords), batch_size):
                batch_cords = patch_cords[i : i + batch_size]
                batch_images = [image.crop((x, y, x + tile_size, y + tile_size)) for x, y in batch_cords]
                batch_features = self(batch_images)
                logger.info(f"Processed batch {i // batch_size + 1} / {(len(patch_cords) + batch_size - 1) // batch_size}")

                for (x, y), tokens in zip(batch_cords, batch_features):
                    # These models follow a ViT architecture, with a patch size of 16.
                    # For a 224x224 image, this results in 1 class token + 4 register tokens + 196 patch tokens = 201 tokens
                    out_x = x // patch_size
                    out_y = y // patch_size
                    features = tokens[1 + num_reg :].reshape(patch_output_size, patch_output_size, -1).cpu()
                    out_features[out_x : out_x + patch_output_size, out_y : out_y + patch_output_size] += features
                    feature_counts[out_x : out_x + patch_output_size, out_y : out_y + patch_output_size] += 1
            
            return out_features / feature_counts.clamp_min(1).unsqueeze(-1)
