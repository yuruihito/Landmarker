#!/bin/bash

set -e

echo "Starting preprocessing..."

poetry run python -m bin.preprocess \
--dataset_dir /mnt/c/Users/kameda/Documents/projects/SwinUNETR/dataset \
--project_name practice_40cases \
--k_fold 4 \
--lm_keys head_center Acetabular_outermost tear_drop \
--sigma 3.0

echo "Preprocessing finished."