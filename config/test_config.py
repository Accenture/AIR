# general I/O parameters
OUTPUT_TYPE = "video"
LABEL_MAPPING = "pascal"
VIDEO_FILE = "data/videos/Ylojarvi-gridiajo-two-guys-moving.mov"
OUT_RESOLUTION = None # (3840, 2024)
OUTPUT_PATH = "data/predictions/Ylojarvi-gridiajo-two-guys-moving-air-output.mov"
FRAME_OFFSET = 200 # 1560
PROCESS_NUM_FRAMES = 300
COMPRESS_VIDEO = True

# detection algorithm parameters
MODEL = "dauntless-sweep-2_resnet152_pascal-mob-inference.h5"
BACKBONE = "resnet152"
DETECT_EVERY_NTH_FRAME = 20
USE_TRACKING = True
SHOW_DETECTION_N_FRAMES = 30
USE_GPU = False
PROFILE = True
IMAGE_TILING_DIM = 2
IMAGE_MIN_SIDE = 1525
IMAGE_MAX_SIDE = 2025

# Results filtering settings
CONFIDENCE_THRES = 0.05
MAX_DETECTIONS_PER_FRAME = 10000

# Bounding box aggregation settings
MERGE_MODE = "enclose"
MOB_ITERS = 3
BBA_IOU_THRES = 0.001
TOP_K=25
