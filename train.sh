#!/bin/bash

set -e

echo "Starting train..."

poetry run python -m bin.train \
--dataset_dir /mnt/c/Users/kameda/Documents/projects/SwinUNETR/dataset \
--project_name practice_40cases \
--output_dir ./workspace \
--patch_size 96 \
--batch_size 2 \
--k_fold 4 \
--lr 1e-4 \
--max_epoch 400 \
--lm_keys head_center Acetabular_outermost tear_drop \

echo "train finished."