# general I/O parameters
OUTPUT_TYPE = "video"
VIDEO_FILE = "../data/videos/Ylojarvi-gridiajo-one-blue-guy-moving.mov"
OUT_RESOLUTION = None # (3840, 2024)
OUTPUT_PATH = None
FRAME_OFFSET = 0 # 1560
PROCESS_NUM_FRAMES = 1000
COMPRESS_VIDEO = True

# algorithm parameters
MODEL = "resnet50_coco_60_inference.h5"
BACKBONE = "resnet50"
BATCH_SIZE = 2
CONFIDENCE_THRES = 0.8
DETECT_EVERY_NTH_FRAME = 360
MAX_DETECTIONS_PER_FRAME = 20
INTERPOLATE_BETWEEN_DETECTIONS = False
SHOW_DETECTION_N_FRAMES = 15
USE_GPU = False
PROFILE = False
