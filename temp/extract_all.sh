#!/bin/bash
FILE_1="intermediate_solutions/main_at_end_step1.py"
mkdir -p $(dirname "$FILE_1")
FILE_2A="intermediate_solutions/main_at_end_step2a.py"
mkdir -p $(dirname "$FILE_2A")
FILE_2B="intermediate_solutions/main_at_end_step2b.py"
mkdir -p $(dirname "$FILE_2B")
FILE_3="intermediate_solutions/main_at_end_step3.py"
mkdir -p $(dirname "$FILE_3")

# Step 1 - preprocessing
bash temp/extract.sh "1-preprocessing.qmd" $FILE_1

# File step2a - GB
bash temp/extract.sh "2-GB_model.qmd" $FILE_2A

# File step2b - RF
bash temp/extract.sh "2-RF_model.qmd" $FILE_2B

# File step3 - metrics
bash temp/extract.sh "3-metrics.qmd" "temp.py" && cat intermediate_solutions/0_fallback.py temp.py > $FILE_3 && rm temp.py

# File step4 - logging
bash temp/extract.sh "3-metrics.qmd" "temp.py" && cat intermediate_solutions/0_fallback.py temp.py > $FILE_3 && rm temp.py
