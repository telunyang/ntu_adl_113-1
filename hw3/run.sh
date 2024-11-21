#!/bin/bash
BASE_MODEL_PATH=$1
ADAPTER_MODEL_PATH=$2
INPUT=$3
OUTPUT=$4

python ./predict.py \
--base_model_path $BASE_MODEL_PATH \
--adapter_model_path $ADAPTER_MODEL_PATH \
--input $INPUT \
--output $OUTPUT