#!/usr/bin/env bash

# Set the output file name
OUTPUT_FILE="combined.txt"

# Check if the output file already exists, and if so, remove it
if [ -f "$OUTPUT_FILE" ]; then
    rm "$OUTPUT_FILE"
fi

# Use a for loop to go through each .rs file and append its content
for file in *.rs; do
    # If there are no .rs files, break out of the loop
    if [ "$file" = "*.rs" ]; then
        echo "No .rs files found."
        exit 1
    fi
    cat "$file" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
done

echo "All .rs files have been merged into $OUTPUT_FILE"
