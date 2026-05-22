#!/bin/bash
# bash solution/admin/extract_all.sh true - to test all scripts
# bash solution/admin/extract_all.sh false - not to test all scripts
TEST=$1

FILE_2="intermediate_solutions/2_preprocessing.py"
mkdir -p $(dirname "$FILE_2")
FILE_3A="intermediate_solutions/3_GB.py"
mkdir -p $(dirname "$FILE_3A")
FILE_3B="intermediate_solutions/3_RF.py"
mkdir -p $(dirname "$FILE_3B")
FILE_4="intermediate_solutions/4_metrics.py"
mkdir -p $(dirname "$FILE_4")

# Step 2 - preprocessing
bash solution/admin/extract.sh "subject/2-preprocessing.qmd" $FILE_2

# File step3a - GB
bash solution/admin/extract.sh "subject/3-GB_model.qmd" $FILE_3A

# File step3b - RF
bash solution/admin/extract.sh "subject/3-RF_model.qmd" $FILE_3B

# File step4 - metrics
bash solution/admin/extract.sh "subject/4-metrics.qmd" $FILE_4 

# Test
if [ "$TEST" = true ]; then
    echo "running $FILE_2"
    uv run "$FILE_2"
    echo "running $FILE_3A"
    uv run "$FILE_3A"
    echo "running $FILE_3B"
    uv run "$FILE_3B"
    echo "running $FILE_4"
    uv run "$FILE_4"
fi