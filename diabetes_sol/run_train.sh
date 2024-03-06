#!/bin/bash

train_dir=./diabetes_data
dev_dir=./diabetes_data
model_dir=./diabetes_save

python main.py train --train_dir "$train_dir" --dev_dir "$dev_dir" --model_dir "$model_dir"
