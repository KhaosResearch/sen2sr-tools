import os
from pathlib import Path

RESOLUTION = 10

CURR_SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = CURR_SCRIPT_DIR / "model"
RGBN_MODEL_DIR = str(MODEL_DIR / "SEN2SRLite_RGBN")
MAIN_MODEL_DIR = str(MODEL_DIR / "SEN2SRLite")

SEN2SR_SR_DIR = Path(
    os.environ.get(
        "SEN2SR_OUTPUT_DIR",
        CURR_SCRIPT_DIR / "sen2sr_out"
    )
)

PNG_DIR = SEN2SR_SR_DIR / "png"
TIF_DIR = SEN2SR_SR_DIR / "tif"

OG_TIF_FILEPATH = TIF_DIR / "original.tif"
SR_TIF_FILEPATH = TIF_DIR / "superres.tif"

OG_PNG_FILEPATH = str(OG_TIF_FILEPATH).replace("tif", "png")
SR_PNG_FILEPATH = str(SR_TIF_FILEPATH).replace("tif", "png")
COMPARISON_PNG_FILEPATH = SEN2SR_SR_DIR / "OG-SR_comparison.png"

GEOJSON_FILEPATH = SEN2SR_SR_DIR / "polygon.geojson"

BANDS = ["B08", "B02", "B03", "B04", "SCL"]  # NIR + RGB + SCL

BRIGHTNESS_FACTOR = 1.2
GAMMA = 0.7
