#!/bin/bash
FILE_2="intermediate_solutions/2_preprocessing.py"
mkdir -p $(dirname "$FILE_2")
FILE_3A="intermediate_solutions/3_GB.py"
mkdir -p $(dirname "$FILE_3A")
FILE_3B="intermediate_solutions/3_RF.py"
mkdir -p $(dirname "$FILE_3B")
FILE_4="intermediate_solutions/4_metrics.py"
mkdir -p $(dirname "$FILE_4")

# Step 1 - preprocessing
bash solution/admin/extract.sh "subject/2-preprocessing.qmd" $FILE_2

# File step2a - GB
bash solution/admin/extract.sh "subject/3-GB_model.qmd" $FILE_3A

# File step2b - RF
bash solution/admin/extract.sh "subject/3-RF_model.qmd" $FILE_3B

# File step3 - metrics
bash solution/admin/extract.sh "subject/4-metrics.qmd" "temp.py" && cat intermediate_solutions/0_fallback.py temp.py > $FILE_4 && rm temp.py
