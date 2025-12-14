import os
from pathlib import Path

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
if os.path.exists("/kaggle/input"):
    print("✓ ENVIRONMENT: KAGGLE")
    INPUT_DIR = Path("/kaggle/input/MABe-mouse-behavior-detection")
    WORKING_DIR = Path("/kaggle/working")
    XGB_MAX_BIN = 256
    NEGATIVE_SAMPLE_RATE = 0.3
else:
    print("✓ ENVIRONMENT: LOCAL")
    INPUT_DIR = Path("D:/BTL_ML_eda/data")
    WORKING_DIR = Path("D:/BTL_ML_eda/output")
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    XGB_MAX_BIN = 64
    NEGATIVE_SAMPLE_RATE = 0.3

TRAIN_TRACKING_DIR = INPUT_DIR / "train_tracking"
TRAIN_ANNOTATION_DIR = INPUT_DIR / "train_annotation"
TEST_TRACKING_DIR = INPUT_DIR / "test_tracking"

# ============================================================================
# INDEX COLUMNS
# ============================================================================
INDEX_COLS = [
    "video_id",
    "agent_mouse_id",
    "target_mouse_id",
    "video_frame",
]

# ============================================================================
# BODY PARTS
# ============================================================================
BODY_PARTS = [
    "ear_left",
    "ear_right",
    "nose",
    "neck",
    "body_center",
    "lateral_left",
    "lateral_right",
    "hip_left",
    "hip_right",
    "tail_base",
    "tail_tip",
]

# ============================================================================
# BEHAVIORS 
# ============================================================================
SELF_BEHAVIORS = [
    "biteobject", "climb", "dig", "exploreobject", "freeze",
    "genitalgroom", "huddle", "rear", "rest", "run", "selfgroom",
]

PAIR_BEHAVIORS = [
    "allogroom", "approach", "attack", "attemptmount", "avoid",
    "chase", "chaseattack", "defend", "disengage", "dominance",
    "dominancegroom", "dominancemount", "ejaculate", "escape",
    "flinch", "follow", "intromit", "mount", "reciprocalsniff",
    "shepherd", "sniff", "sniffbody", "sniffface", "sniffgenital",
    "submit", "tussle",
]

ALL_BEHAVIORS = SELF_BEHAVIORS + PAIR_BEHAVIORS

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================
ROLLING_WINDOWS = [5, 15, 30, 60, 90, 120]  # frames

# Body part pairs để tính distance trong pair features
BODY_PART_PAIRS_FOR_DISTANCE = [
    ("nose", "nose"),
    ("nose", "tail_base"),
    ("body_center", "body_center"),
]

# Periods cho speed calculation
SPEED_PERIODS_MS = [500, 1000, 2000, 3000]

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
N_FOLDS = 3
RANDOM_STATE = 42

# XGBoost parameters cho hành vi hiếm 
RARE_BEHAVIOR_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 10,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "gamma": 1.0,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "seed": RANDOM_STATE,
}

# XGBoost parameters cho hành vi phổ biến 
COMMON_BEHAVIOR_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.5,
    "seed": RANDOM_STATE,
}

# Số rounds training
RARE_BEHAVIOR_ROUNDS = 300
COMMON_BEHAVIOR_ROUNDS = 500
RARE_EARLY_STOP = 15
COMMON_EARLY_STOP = 20

# Ngưỡng để xác định hành vi hiếm
RARE_BEHAVIOR_THRESHOLD = 0.001  # < 0.1% samples

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================
SELF_FEATURES_DIR = WORKING_DIR / "self_features"
PAIR_FEATURES_DIR = WORKING_DIR / "pair_features"
RESULTS_DIR = WORKING_DIR / "results"

for dir_path in [SELF_FEATURES_DIR, PAIR_FEATURES_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# ============================================================================
# LOGGING
# ============================================================================
VERBOSE = True
LOG_INTERVAL = 10