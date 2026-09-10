import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path("w:/Project Work/SIH")
DATASET_PATH = BASE_DIR / "models" / "prototype_land_acquisition_cases.csv"
OUTPUT_IPYNB = BASE_DIR / "models" / "Experimental_Data_of_Model.ipynb"
OUTPUT_PY = BASE_DIR / "models" / "Experimental_Data_of_Model.py"

print(f"Dataset exists: {DATASET_PATH.exists()}")
