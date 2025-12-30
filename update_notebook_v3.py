import json
import os

b_path = 'voronoi_structure_analysis.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. New Data Loading Code with Particle Count Search
new_data_load_code = [
    "# Dataset Path\n",
    