#!/bin/bash
# bash temp/extract.sh "2-GB_model.qmd" "intermediate_solutions/solution_step1_to_2a/main.py"
# bash temp/extract.sh "2-RF_model.qmd" "intermediate_solutions/solution_step1_to_2b/main.py"
# bash temp/extract.sh "1-preprocessing.qmd" "intermediate_solutions/solution_step1/main.py"

# Check if input file is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_file.qmd>"
    exit 1
fi

input_file="$1"

if [ "$#" -ne 2 ]; then
    output_file="${input_file%.qmd}.py"
else
    output_file="$2"
    folder_output=$(dirname "$output_file")
    mkdir -p "$folder_output" && echo "Folder '$folder_output' created."
fi



# Extract Python code blocks, remove #| lines, and trim the exact number of leading whitespace
awk '
/^[[:space:]]*```\{python/ {
    leading_ws = 0;
    temp = $0;
    while (substr(temp, leading_ws + 1, 1) ~ /[[:space:]]/) {
        leading_ws++;
    }
    flag = 1;
    next;
}
/^[[:space:]]*```/ && flag {
    flag = 0;
    next;
}
flag {
    if (leading_ws > 0 && length($0) >= leading_ws) {
        $0 = substr($0, leading_ws + 1);
    }
    if (!/^[[:space:]]*#\|/) {
        print;
    }
}
' "$input_file" > "$output_file"

echo "Python code blocks extracted from $input_file ==> $output_file"
