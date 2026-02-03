import seaborn

PALETTE_NAME = "pastel"
PALETTE = seaborn.color_palette(PALETTE_NAME, 100)
# Support both 'val' and 'valid' split namings used across the codebase/tests
LABELS_PALETTE = {"train": PALETTE[0], "val": PALETTE[1], "valid": PALETTE[1], "test": PALETTE[2]}
