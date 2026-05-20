#!/bin/bash


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
fi



# Extract Python code blocks, remove #| lines, and trim the exact number of leading whitespace
awk '
/# Exercice/ {
    print ""; print $0;
    next;
    }
/^[[:space:]]*```\{python/ {
    print "# %%"
    leading_ws = 0;
    temp = $0;
    while (substr(temp, leading_ws + 1, 1) ~ /[[:space:]]/) {
        leading_ws++;
    }
    flag = 1;
    next;
}
/^[[:space:]]*```/ && flag {
    flag = 0; print ""
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
