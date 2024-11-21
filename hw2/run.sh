#!/bin/bash
INPUT=$1
OUTPUT=$2

python ./predict_t5.py \
--input $INPUT \
--output $OUTPUT

# python ./predict_gpt2.py \
# --input $INPUT \
# --output $OUTPUT