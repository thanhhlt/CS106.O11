#!/bin/bash

train_dir=./compas_data
dev_dir=./compas_data
model_dir=./compas_save

python main.py train --train_dir "$train_dir" --dev_dir "$dev_dir" --model_dir "$model_dir"
