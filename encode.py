#!/usr/bin/env python3
import sys
import numpy as np
from scipy.io import wavfile
import struct
import lzma
import warnings

# --- Configuration ---
# Set to False to run normal encoding for eval.sh
RUN_ANALYSIS = False
# File to analyze if RUN_ANALYSIS is True (path relative to script location)
# This path is only used if RUN_ANALYSIS is True
ANALYSIS_FILE_PATH = 'data/7008d9c8-6868-47eb-9935-3cf6885cdb1d.wav'

# Suppress harmless WavFileWarning
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

def run_normal_encoding(input_wav, output_bw):
    """Encodes a single WAV file using Delta (stored as int32) + LZMA."""
    try:
        # 1. Read WAV file
        sample_rate, data = wavfile.read(input_wav)

        # Ensure input data is int16 as expected from WAV
        if data.dtype != np.int16:
            print(f"Error: Expected int16 data in {input_wav}, got {data.dtype}", file=sys.stderr)
            sys.exit(1)

        # Handle empty input file
        if len(data) == 0:
             print(f"Warning: Input file {input_wav} is empty.", file=sys.stderr)
             first_sample = 0 # Assign a default value for empty data
             # Create an empty array with dtype int32 for consistency
             deltas_int32 = np.array([], dtype=np.int32)
             # Compress the (empty) int32 delta array bytes
             compressed_deltas = lzma.compress(deltas_int32.tobytes())
        else:
            # 2. Perform Delta Coding
            # Store the first sample (still int16)
            first_sample = data[0]
            # Calculate differences using int32. These will now fit any jump.
            deltas_int32 = np.diff(data.astype(np.int32))

            # 3. Compress the int32 Deltas using LZMA
            # Compressing 4 bytes per delta instead of 2 will impact ratio
            compressed_deltas = lzma.compress(deltas_int32.tobytes())

        # 4. Write Output File
        with open(output_bw, 'wb') as f:
            # Metadata: sample rate (as uint32), first sample (as int16)
            # '<I' = little-endian unsigned int (4 bytes) for sample rate
            # '<h' = little-endian short (2 bytes) for int16 first sample
            metadata = struct.pack('<Ih', sample_rate, first_sample)
            f.write(metadata)
            # Write the compressed stream of int32 delta bytes
            f.write(compressed_deltas)

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_wav}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Print error specific to the file being processed
        print(f"An error occurred during encoding {input_wav}: {e}", file=sys.stderr)
        # Exit because eval.sh expects successful processing of all files or failure
        sys.exit(1)

def run_analysis(input_wav):
    """Analyzes delta distribution for a single WAV file (requires matplotlib).
       Note: This function would need updates to properly analyze int32 deltas if used.
    """
    try:
        # Dynamically import matplotlib only when needed for analysis
        import matplotlib.pyplot as plt
        print(f"--- Running analysis on: {input_wav} ---")
        print("Note: Analysis function may need updates for int32 deltas.")
        sample_rate, data = wavfile.read(input_wav)

        if data.dtype != np.int16:
            print(f"Error: Expected int16 data, got {data.dtype}", file=sys.stderr)
            sys.exit(1)
        if len(data) == 0:
            print(f"Input file {input_wav} is empty.")
            sys.exit(0)

        # Calculate deltas as int32
        deltas_int32 = np.diff(data.astype(np.int32))

        print("\nDelta Statistics (int32):") # Label as int32
        min_d, max_d = np.min(deltas_int32), np.max(deltas_int32)
        mean_d, std_d = np.mean(deltas_int32), np.std(deltas_int32)
        zeros = np.sum(deltas_int32 == 0)
        total_deltas = len(deltas_int32)
        zero_pct = zeros * 100.0 / total_deltas if total_deltas > 0 else 0

        print(f"  Min: {min_d}")
        print(f"  Max: {max_d}")
        print(f"  Mean: {mean_d:.2f}")
        print(f"  Std Dev: {std_d:.2f}")
        print(f"  Zero Deltas: {zeros} / {total_deltas} ({zero_pct:.2f}%)")

        # Plot Histogram - Range might need significant adjustment for int32
        print("\nGenerating plot (range may need adjustment)...")
        plt.figure(figsize=(12, 6))
        # Determine a suitable plot range based on observed min/max for int32
        plot_range = (-max(abs(min_d), abs(max_d), 500), max(abs(min_d), abs(max_d), 500)) # Dynamic range example
        plot_range = (-2000, 2000) # Or keep fixed if preferred, e.g. based on previous int16 range

        counts, bins, patches = plt.hist(deltas_int32, bins=200, range=plot_range)

        plt.title(f'Histogram of Delta Values (int32) for {input_wav}\n(Range {plot_range[0]} to {plot_range[1]})')
        plt.xlabel('Delta Value (int32)')
        plt.ylabel('Frequency')
        plt.grid(True)
        stats_text = (f"Min: {min_d}, Max: {max_d}\n"
                      f"Mean: {mean_d:.2f}, StdDev: {std_d:.2f}\n"
                      f"Zeros: {zeros} ({zero_pct:.2f}%)")
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        print("\nClose the plot window to exit.")
        plt.show()

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_wav}", file=sys.stderr)
        sys.exit(1)
    except ModuleNotFoundError:
         print("\nError: Matplotlib not found. Please install it (`pip install matplotlib`) to run analysis.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"An error occurred during analysis: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main execution logic: either run analysis or normal encoding."""
    if RUN_ANALYSIS:
        # Check for matplotlib before attempting analysis
        try:
            import matplotlib
        except ModuleNotFoundError:
             print("\nError: Matplotlib not found. Please install it (`pip install matplotlib`) to run analysis.", file=sys.stderr)
             sys.exit(1)
        run_analysis(ANALYSIS_FILE_PATH)
    else:
        # Normal operation: expect command-line args for input/output files
        if len(sys.argv) != 3:
            print(f"Usage (normal mode): {sys.argv[0]} <input_wav_file> <output_brainwire_file>")
            sys.exit(1)
        input_wav = sys.argv[1]
        output_bw = sys.argv[2]
        run_normal_encoding(input_wav, output_bw)

if __name__ == "__main__":
    main()