"""Shared configuration for Bretez processing."""

# The Turgot map is about 25004 / (651838 - 650005) ~= 13 px/m, while the
# training imagery is 0.6 m/px. Downscaling by 8 brings the map close to the
# training-set ground resolution before feature extraction.
INPUT_IMAGE_DOWNSCALE_FACTOR = 4
DINO_PATCH_SIZE = 16
MAP_PIXELS_PER_FEATURE = INPUT_IMAGE_DOWNSCALE_FACTOR * DINO_PATCH_SIZE
