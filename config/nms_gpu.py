# general I/O parameters
OUTPUT_TYPE = "video"
LABEL_MAPPING = "pascal"
VIDEO_FILE = "data/videos/Ylojarvi-gridiajo-two-guys-moving.mov"
OUT_RESOLUTION = None # (3840, 2024)
OUTPUT_PATH = "data/predictions/Ylojarvi-gridiajo-two-guys-moving-air-output.mov"
FRAME_OFFSET = 0 # 1560
PROCESS_NUM_FRAMES = None
COMPRESS_VIDEO = True

# detection algorithm parameters
MODEL = "dauntless-sweep-2_resnet152_pascal-nms-inference.h5"
BACKBONE = "resnet152"
DETECT_EVERY_NTH_FRAME = 20
USE_TRACKING = False
PLOT_OBJECT_SPEED = False
SHOW_DETECTION_N_FRAMES = 30
USE_GPU = True
PROFILE = False
IMAGE_TILING_DIM = 2
IMAGE_MIN_SIDE = 1525
IMAGE_MAX_SIDE = 2025

# Results filtering settings
CONFIDENCE_THRES = 0.25
MAX_DETECTIONS_PER_FRAME = 1000

# Bounding box aggregation settings
MERGE_MODE = "argmax"
MOB_ITERS = 1
BBA_IOU_THRES = 0.5
TOP_K=-1
