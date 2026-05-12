#!/bin/bash
FILE_1="intermediate_solutions/solution_step1/main.py"
mkdir -p $(dirname "$FILE_1")
FILE_2A="intermediate_solutions/solution_step1_to_2a/main.py"
mkdir -p $(dirname "$FILE_2A")
FILE_2B="intermediate_solutions/solution_step1_to_2b/main.py"
mkdir -p $(dirname "$FILE_2B")
FILE_3="intermediate_solutions/solution_step1_to_3/main.py"
mkdir -p $(dirname "$FILE_3")

# Step 1 - preprocessing
bash temp/extract.sh "1-preprocessing.qmd" $FILE_1

# File step2a - GB
bash temp/extract.sh "2-GB_model.qmd" $FILE_2A

# File step2b - RF
bash temp/extract.sh "2-RF_model.qmd" "temp.py" && cat $FILE_1 temp.py > $FILE_2B && rm temp.py

# File step3 - metrics
bash temp/extract.sh "3-metrics.qmd" "temp.py" && cat starting_point/0_fallback.py temp.py > $FILE_3 && rm temp.py

# File step4 - logging
bash temp/extract.sh "3-metrics.qmd" "temp.py" && cat starting_point/0_fallback.py temp.py > $FILE_3 && rm temp.py
