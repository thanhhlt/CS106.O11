#!/bin/bash

model_dir=./compas_save
input_dir=$2
output_path=$3

python main.py predict --model_dir "$model_dir" --input_dir "$input_dir" --output_path "$output_path"
