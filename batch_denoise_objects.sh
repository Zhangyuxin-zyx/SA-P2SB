#!/bin/bash

INPUT_DIR=""

OUTPUT_DIR=""

MODEL_PATH=""

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.xyz
do
    filename=$(basename "$file")

    output_file="$OUTPUT_DIR/$filename"

    echo "Processing: $file"
    
    python denoise_object.py \
        --data_path "$file" \
        --save_path "$output_file" \
        --model_path "$MODEL_PATH"

done

echo "All files processed."
