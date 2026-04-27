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

    def process_image(self, image: Image.Image) -> torch.Tensor:
        """Process the full image in one model call when it fits, otherwise tile it."""
        try:
            return self.process_image_one_pass(image)
        except torch.cuda.OutOfMemoryError:
            logger.warning("One-pass processing ran out of CUDA memory; falling back to tiled processing.")
            torch.cuda.empty_cache()
            return self.process_big_image(image)

    def process_image_one_pass(self, image: Image.Image) -> torch.Tensor:
        logger.info(f"Processing image in one pass with size {image.size}")
        patch_size = self.model.config.patch_size
        num_reg = getattr(self.model.config, "num_register_tokens", 0)
        with torch.inference_mode():
            inputs = self.processor(images=image, return_tensors="pt", do_resize=False).to("cuda")
            pixel_height, pixel_width = inputs["pixel_values"].shape[-2:]
            feature_width = pixel_width // patch_size
            feature_height = pixel_height // patch_size
            outputs = self.model(**inputs)
            tokens = outputs.last_hidden_state[0]
            features = tokens[1 + num_reg :].reshape(feature_height, feature_width, -1).cpu()
            return features.transpose(0, 1).contiguous()

    def process_big_image(self, image: Image.Image) -> torch.Tensor: 
        "splits the image into overlapping patches and center-weight averages the features"
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
        patch_positions = torch.arange(patch_output_size, dtype=torch.float32)
        feature_weights_1d = torch.minimum(patch_positions + 1, patch_output_size - patch_positions)
        feature_weights = torch.outer(feature_weights_1d, feature_weights_1d)
        with torch.inference_mode():
            out_features = torch.zeros(
                (
                    (width_patches - 1) * stride_output_size + patch_output_size,
                    (height_patches - 1) * stride_output_size + patch_output_size,
                    self.model.config.hidden_size,
                )
            )
            out_feature_weights = torch.zeros(out_features.shape[:2], dtype=out_features.dtype)
            logger.debug(f"Processing image of size {image.size} with patch size {tile_size} and stride {stride}")
            patch_cords = list(
                itertools.product(
                    range(0, width - tile_size + 1, stride),
                    range(0, height - tile_size + 1, stride),
                )
            )
            # batch and process patches
            batch_size = 8
            
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
                    features = tokens[1 + num_reg :].reshape(patch_output_size, patch_output_size, -1).transpose(0, 1).cpu()
                    feature_slice = (slice(out_x, out_x + patch_output_size), slice(out_y, out_y + patch_output_size))
                    out_features[feature_slice] += features * feature_weights.unsqueeze(-1)
                    out_feature_weights[feature_slice] += feature_weights
            
            return out_features / out_feature_weights.clamp_min(1).unsqueeze(-1)
