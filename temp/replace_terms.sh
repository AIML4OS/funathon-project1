#!/bin/bash

# Usage: ./replace_terms_in_paths.sh /path/to/paths.txt /path/to/replacements.json

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install it first."
    exit 1
fi

# Inputs
PATHS_FILE="$1"
REPLACEMENTS_JSON="$2"

# Check if inputs are provided
if [[ -z "$PATHS_FILE" || -z "$REPLACEMENTS_JSON" ]]; then
    echo "Usage: $0 /path/to/paths.txt /path/to/replacements.json"
    exit 1
fi

# Read replacements from JSON
REPLACEMENTS=$(jq -r 'to_entries[] | "\(.key)|\(.value)"' "$REPLACEMENTS_JSON")

# Read paths from the .txt file
while IFS= read -r path; do
    # Skip empty lines
    if [[ -z "$path" ]]; then
        continue
    fi

    # Check if the path exists
    if [[ ! -e "$path" ]]; then
        echo "Warning: Path does not exist: $path"
        continue
    fi

    # Find all files in the path (recursively if it's a directory)
    find "$path" -type f | while read -r file; do
        # Backup the file
        # cp "$file" "$file.bak"

        # Apply all replacements
        while IFS='|' read -r term replacement; do
            # Escape special characters for sed
            # escaped_term=$(printf '%s\n' "$term" | sed -e 's/[\/&]/\\&/g')
            # escaped_replacement=$(printf '%s\n' "$replacement" | sed -e 's/[\/&]/\\&/g')

            # Replace in file
            sed -i "s/$term/$replacement/g" "$file"
        done <<< "$REPLACEMENTS"

        echo "Processed: $file"
    done
done < "$PATHS_FILE"

echo "Replacement complete."