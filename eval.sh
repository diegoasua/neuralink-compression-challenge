#!/usr/bin/env bash

# Check if data.zip exists
if [ ! -f data.zip ]; then
    echo "Error: data.zip not found"
    exit 1
fi

rm -rf data
unzip data.zip

get_file_size() {
    # More portable way to get file size
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        stat -f %z "$1"
    else
        # Linux
        stat -c %s "$1"
    fi
}

total_size_raw=0
encoder_size=$(get_file_size encode.py)
decoder_size=$(get_file_size decode.py)
total_size_compressed=$((encoder_size + decoder_size))

# Make sure Python scripts are executable
chmod +x encode.py decode.py

# Check if data directory exists and has files
if [ ! -d data ] || [ -z "$(ls -A data)" ]; then
    echo "Error: data directory is empty or doesn't exist"
    exit 1
fi

for file in data/*.wav; do
    if [ ! -f "$file" ]; then
        echo "No WAV files found in data directory"
        exit 1
    fi
    
    echo "Processing $file"
    compressed_file_path="${file}.brainwire"
    decompressed_file_path="${file}.copy"

    ./encode.py "$file" "$compressed_file_path"
    ./decode.py "$compressed_file_path" "$decompressed_file_path"

    file_size=$(get_file_size "$file")
    compressed_size=$(get_file_size "$compressed_file_path")

    if diff -q "$file" "$decompressed_file_path" > /dev/null; then
        echo "${file} losslessly compressed from ${file_size} bytes to ${compressed_size} bytes"
    else
        echo "ERROR: ${file} and ${decompressed_file_path} are different."
        exit 1
    fi

    total_size_raw=$((total_size_raw + file_size))
    total_size_compressed=$((total_size_compressed + compressed_size))
done

compression_ratio=$(echo "scale=2; ${total_size_raw} / ${total_size_compressed}" | bc)

echo "All recordings successfully compressed."
echo "Original size (bytes): ${total_size_raw}"
echo "Compressed size (bytes): ${total_size_compressed}"
echo "Compression ratio: ${compression_ratio}"