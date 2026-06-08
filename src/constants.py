import sys
import torch

from datetime import datetime
from huggingface_hub import snapshot_download
from pathlib import Path


# ee chipping configs
EE_PROJECT = 'ee-myscon'
PIXEL_SIZE = 30
CHIP_SIZE = 224
PARALLELLISM = 16
EPSG_CODE = 'EPSG:5070'
COMPOSITE_DATE_START = datetime(2000, 7, 15)
COMPOSITE_DATE_END = datetime(2000, 8, 31)
SCALE = .0001

# data directories
DATA_DIR = Path("/hudak_agb/data")
IMAGERY_DIR = DATA_DIR / "pnw"
PRISM_RPR_DIR = DATA_DIR / "prism_reprojected"
PRISM_ZIP_DIR = DATA_DIR / "prism_monthly"
SOLUS_DIR = DATA_DIR / "solus100"
STAT_PATH = DATA_DIR / "stat.yml"

TRAIN_OUT_DIR = DATA_DIR / 'outputs'
UNITS_DIR = DATA_DIR / 'Forest_AGB_NW_USA_V2_2443/data/'
UNITS_PATH = UNITS_DIR / 'Forest_AGB_NW_LidarUnits.zip'
PLOTS_DIR = DATA_DIR / "plot_data"
GRID_PLOTS_PATH = DATA_DIR / "hudak_agb_grid_invy.geojson"

# imagery directories
SENSOR_DIRS = [DATA_DIR / 'LS', DATA_DIR / 'HLSL', DATA_DIR / 'HLSS', DATA_DIR / 'HLS', DATA_DIR / 'GLO30', DATA_DIR / 'NASADEM']
SENSOR_YEARS = [range(2000, 2017), range(2013, 2026), range(2016, 2026), range(2016, 2026), None, None]    
DIRS_YEARS = {p.name: y for p, y in zip(SENSOR_DIRS, SENSOR_YEARS)}
PRISM_YEARS = list(range(1991, 2021))

# bands elements
PRISM_ELEMENTS = ['ppt', 'tdmean', 'tmax', 'tmean', 'tmin', 'vpdmax', 'vpdmin']
LBANDS = ["Fmask", "B2", "B3", "B4", "B5", "B6", "B7"]
SBANDS = ["Fmask", "B2", "B3", "B4", "B8A", "B11", "B12"]
BANDS = LBANDS[1:]

# file paths
PRISM_FILES = {}
for e in PRISM_ELEMENTS:
    PRISM_FILES[e] = sorted(list(PRISM_RPR_DIR.glob(f"*{e}*.tif")))
SOLUS_FILES = sorted(list(SOLUS_DIR.glob(f"*.tif")))

PLOT_GRID = DATA_DIR / 'hudak_agb_grid_buff.geojson'
FISHNET_GRID = DATA_DIR / 'hudak_agb_grid.geojson'
CLIP_GRID = DATA_DIR / 'hudak_agb_grid_clip.geojson'

# downloading repo from huggingface w/ checkpoint and architecture code
PRITHVI_DIR = snapshot_download(repo_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M")
sys.path.append(PRITHVI_DIR)

# repo path
RP_DIR = Path(PRITHVI_DIR)
PT_PATH = RP_DIR / 'Prithvi_EO_V2_300M.pt'
CF_PATH = RP_DIR / 'config.json'

# model inference configs
PATCH_SIZE = 16
L_SIZE = CHIP_SIZE // PATCH_SIZE
TOP_CKPT = 4

DEVICE = torch.device('cuda')
DTYPE = torch.bfloat16
CDTYPE = torch.cfloat
NO_DATA = -9999.0
INF_NO_DATA = 65535

GRID_SIZE = PIXEL_SIZE * CHIP_SIZE
CLIP_SIZE = GRID_SIZE // 4
NUM_JITTERS = 32
BATCH_SIZE = 32
