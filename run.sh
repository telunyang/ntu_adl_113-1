#!/bin/bash
CONTEXT_PATH=$1
TEST_PATH=$2
OUTPUT_CSV_PATH=$3

python ./run.py \
--context_path $CONTEXT_PATH \
--test_path $TEST_PATH \
--output_csv_path $OUTPUT_CSV_PATH