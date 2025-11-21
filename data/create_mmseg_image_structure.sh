#!/bin/bash

# --- Configuration ---
TARGET_DIMENSION="1024 x 1024" # The dimension string to look for

# --- Argument Handling ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <image_directory> <output_filename>"
    echo "Example: $0 ./images matching_files.txt"
    exit 1
fi

IMAGE_DIR="$1"
OUTPUT_FILE="$2"

# Check if image directory exists and is a directory
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory '$IMAGE_DIR' not found or is not a directory."
    exit 1
fi

# Ensure output file directory exists (optional, but good practice)
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Warning: Output directory '$OUTPUT_DIR' does not exist. Creating it."
    mkdir -p "$OUTPUT_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create output directory '$OUTPUT_DIR'."
        exit 1
    fi
fi


# --- Initialization ---
total_png_count=0
matching_dimension_count=0

# Clear the output file before starting
> "$OUTPUT_FILE"
echo "Cleared/created output file: $OUTPUT_FILE"


# --- Processing ---
echo "Scanning directory: $IMAGE_DIR"

# Use find to reliably handle filenames with spaces or special characters
# -print0 and read -d $'\0' ensure safe handling
find "$IMAGE_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.PNG" \) -print0 | while IFS= read -r -d $'\0' file; do
    ((total_png_count++))

    # Run the 'file' command and capture its output
    file_info=$(file "$file")

    # Check if the output contains the target dimension string
    # Add comma to make match more specific to the dimension part
    if echo "$file_info" | grep -q ", ${TARGET_DIMENSION},"; then
        ((matching_dimension_count++))

        # Extract filename from path
        filename=$(basename "$file")
        # Extract name without .png extension (case-insensitive)
        base_name=$(echo "$filename" | sed -E 's/\.png$//i')

        # Append the base name to the output file
        echo "$base_name" >> "$OUTPUT_FILE"
    fi

    # Optional: Print progress indicator every 100 files
    if (( total_png_count % 100 == 0 )); then
        echo -n "." # Print a dot for progress
    fi
done

echo # Newline after progress dots

# --- Calculation and Output ---
echo "Scan complete."

exit 0