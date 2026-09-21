"""
OceanEmbed — central configuration.
Every other script imports constants from here so there is one source of truth.
"""

# ---------------- Domain ----------------
LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
GRID_RES = 0.25  # degrees

# ---------------- Standard depths (meters) ----------------
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# ---------------- Temporal ----------------
TIME_WINDOW = 14          # days of surface history fed per prediction
TRAIN_YEARS = list(range(2015, 2019))   # 2015-2018
VAL_YEARS = [2019]
TEST_YEARS = [2020]
PRETRAIN_YEARS = list(range(2021, 2024))  # 2021-2023, non-overlapping with Stage 2

# ---------------- Channels ----------------
SURFACE_VARS = ["sst", "sss", "sla", "u_curr", "v_curr", "u_wind", "v_wind"]  # 7
COORD_CHANNELS = ["lat", "lon", "bathy"]                                      # 3
SEASONAL_CHANNELS = ["doy_sin", "doy_cos"]                                    # 2
VALIDITY_CHANNELS = [f"valid_{v}" for v in SURFACE_VARS]                      # 7
ALL_CHANNELS = SURFACE_VARS + COORD_CHANNELS + SEASONAL_CHANNELS + VALIDITY_CHANNELS
N_CHANNELS = len(ALL_CHANNELS)  # 19

# ---------------- Paths ----------------
RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"
MASK_DIR = "./data/masks"
CLIMATOLOGY_PATH = "./data/climatology.nc"
CHECKPOINT_DIR = "./checkpoints"

# ---------------- Model ----------------
ENCODER_BASE_CH = 32
FOURIER_N_FREQS = 8
DEPTH_FEAT_DIM = 32
DEPTH_MLP_HIDDEN = 64

# ---------------- Training ----------------
BATCH_SIZE = 4
STAGE1_EPOCHS = 30
STAGE2_EPOCHS = 50
LR_STAGE1 = 1e-3
LR_STAGE2 = 3e-4
LAMBDA_SMOOTH = 0.1
LAMBDA_GRAD = 0.1
LAMBDA_THERMO = 0.5
THERMO_BAND = (75, 150)   # meters, bounded upweighting zone
THERMO_WEIGHT = 1.5

# ---------------- Ensembling ----------------
N_ENSEMBLE = 3
MC_DROPOUT_PASSES = 20
