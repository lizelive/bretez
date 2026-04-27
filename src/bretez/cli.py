import logging

import torch

from bretez.backbone import Backbone

from .loader import load_image

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(filename='bretez.log', level=logging.DEBUG)

    img = load_image()
    logger.info(f"Image size: {img.size}")
    model = Backbone()
    logger.debug("Processing image...")
    data = model.process_image(img)
    torch.save(data, "features.pt")