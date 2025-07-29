import numpy as np
# MEANs
#IMAGE_MEANS =np.array([117.67, 130.39, 121.52, 162.92]) # Chesapeake
IMAGE_MEANS = np.array([106.13, 110.64, 95.32]) # Poland
# IMAGE_MEANS = np.array([108.79, 113.64, 106.78]) # NRW

# STDs
#IMAGE_STDS = np.array([39.25,37.82,24.24,60.03]) # Chesapeake
IMAGE_STDS = np.array([45.93, 38.80, 33.04]) # Poland
# IMAGE_STDS = np.array([49.52, 43.63, 41.11]) # NRW

# LABEL CLASSES
#LABEL_CLASSES = [0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95] # Chesapeake
LABEL_CLASSES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100] # ESA_GLC10 (NRW, Poland)

# COLORMAPs

LABEL_CLASS_COLORMAP = { # Colormap for ESA_GLC10 dataset
    0:  (0, 0, 0),
    10: (0, 100, 0),
    20: (255, 187, 34),
    30: (255, 255, 76),
    40: (240, 150, 255),
    50: (255, 0, 0),
    60: (180, 180, 180),
    70: (240, 240, 240),
    80: (0, 100, 200),
    90: (0, 150, 160),
    95: (0, 207, 117),
    100: (250, 230, 160),
}

'''
LABEL_CLASS_COLORMAP = { # Color map for Chesapeake dataset
    0:  (0, 0, 0),
    11: (70, 107, 159),
    12: (209, 222, 248),
    21: (222, 197, 197),
    22: (217, 146, 130),
    23: (235, 0, 0),
    24: (171, 0, 0),
    31: (179, 172, 159),
    41: (104, 171, 95),
    42: (28, 95, 44),
    43: (181, 197, 143),
    52: (204, 184, 121),
    71: (223, 223, 194),
    81: (220, 217, 57),
    82: (171, 108, 40),
    90: (184, 217, 235),
    95: (108, 159, 184)
}
'''

# Pred
IDX_TO_EVAL_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 2,
    4: 2,
    5: 0,
    6: 2,
    7: 2,
    8: 3,
    9: 2,
    10: 1,
    11: 2
}

# GT
GT_TO_EVAL_MAP = {
    0: 4,
    1: 2,
    2: 2,
    3: 0,
    4: 0,
    5: 1,
    6: 3,
    7: 2,
    8: 0,
}

CLASS_COLORMAP_MIOU = {
    0: (255, 0, 0),     # Built-up
    1: (0, 100, 0),     # Tree cover
    2: (255, 255, 76),  # Low vegetation
    3: (0, 100, 200),   # Water
    #4: (0, 0, 0)       # Nothing (will be ignored in mIoU calculation)
}

NUM_EVAL_CLASSES = 4

LABEL_IDX_COLORMAP = {
    idx: LABEL_CLASS_COLORMAP[c]
    for idx, c in enumerate(LABEL_CLASSES)
}

def get_label_class_to_idx_map():
    label_to_idx_map = []
    idx = 0
    for i in range(LABEL_CLASSES[-1]+1):
        if i in LABEL_CLASSES:
            label_to_idx_map.append(idx)
            idx += 1
        else:
            label_to_idx_map.append(0)
    label_to_idx_map = np.array(label_to_idx_map).astype(np.int64)
    return label_to_idx_map

LABEL_CLASS_TO_IDX_MAP = get_label_class_to_idx_map()