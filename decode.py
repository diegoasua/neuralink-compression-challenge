#!/usr/bin/env python3
import sys
import numpy as np
from scipy.io import wavfile
import struct
import lzma
import warnings

# Suppress harmless WavFileWarning
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

def main():
    """Main function to decode .brainwire file (containing int32 deltas) to WAV."""
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_brainwire_file> <output_wav_file>")
        sys.exit(1)

    input_bw = sys.argv[1]
    output_wav = sys.argv[2]

    try:
        with open(input_bw, 'rb') as f:
            # 1. Read metadata: sample rate (uint32), first sample (int16)
            # Metadata size = 4 bytes (uint32) + 2 bytes (int16) = 6 bytes
            metadata_bytes = f.read(6)
            if len(metadata_bytes) < 6:
                 print(f"Error: Could not read complete metadata (6 bytes) from {input_bw}", file=sys.stderr)
                 sys.exit(1)
            # Unpack as little-endian unsigned int ('<I') and short ('<h')
            sample_rate, first_sample = struct.unpack('<Ih', metadata_bytes)

            # 2. Read the rest of the data (compressed int32 deltas)
            compressed_deltas = f.read()

        # 3. Decompress Deltas using LZMA (expecting bytes representing int32 values)
        if not compressed_deltas:
             # Handle case where the compressed data stream is empty
             # (e.g., if the original WAV file was empty)
             delta_bytes = b''
        else:
             delta_bytes = lzma.decompress(compressed_deltas)

        # Convert delta bytes back to int32 numpy array
        # Ensure the byte stream length is a multiple of 4 (size of int32)
        if len(delta_bytes) % 4 != 0:
             print(f"Error: Corrupted delta stream in {input_bw}, length {len(delta_bytes)} is not multiple of 4 bytes for int32", file=sys.stderr)
             sys.exit(1)
        # Interpret the bytes as little-endian int32 values
        deltas_int32 = np.frombuffer(delta_bytes, dtype=np.int32)

        # 4. Reconstruct Original Data from Deltas
        # Determine the number of samples in the original file
        num_samples = len(deltas_int32) + 1
        # Special case: if delta_bytes was empty, original file was empty or had 1 sample.
        # If delta_bytes is empty AND num_samples is 1, it means original had 1 sample.
        # If delta_bytes is empty AND first_sample was default (e.g., 0 from empty encode), original was empty.
        # Let's refine the empty check based on byte length:
        if len(delta_bytes) == 0:
            if len(metadata_bytes) == 6 : # We read metadata, so potentially 1 sample
                 num_samples = 1
            else: # Should not happen if metadata read succeeded, but safer
                 num_samples = 0


        # Use int64 for the accumulator during reconstruction to prevent overflow,
        # even though the final result must fit in int16.
        reconstructed_data = np.zeros(num_samples, dtype=np.int64)

        if num_samples > 0:
            # Set the first sample
            reconstructed_data[0] = first_sample
            # Reconstruct the rest using cumulative sum on the int32 deltas
            np.cumsum(deltas_int32, out=reconstructed_data[1:])
            reconstructed_data[1:] += first_sample

        # 5. Final Check and Conversion for Saving
        # CRITICAL: Ensure the fully reconstructed data fits within the int16 range
        # before converting and saving. This verifies losslessness.
        if np.any(reconstructed_data > 32767) or np.any(reconstructed_data < -32768):
             # If this check fails, something is wrong - either the input WAV
             # wasn't truly int16, or an error occurred in processing.
             print(f"Error: Reconstructed data exceeds int16 range in {input_bw}. Lossy result or invalid input data.", file=sys.stderr)
             sys.exit(1)
        # Convert the final reconstructed data to int16 for writing to WAV
        final_data = reconstructed_data.astype(np.int16)


        # 6. Write WAV file
        # Ensure the sample rate read from metadata is valid
        if sample_rate <= 0:
            print(f"Error: Invalid sample rate ({sample_rate}) read from {input_bw}", file=sys.stderr)
            sys.exit(1)
        # Write the reconstructed int16 data as a WAV file
        wavfile.write(output_wav, sample_rate, final_data)

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_bw}", file=sys.stderr)
        sys.exit(1)
    except lzma.LZMAError as e: # Catch LZMA specific decompression errors
        print(f"Error: LZMA decompression failed for {input_bw}: {e}", file=sys.stderr)
        sys.exit(1)
    except struct.error as e:
        # Error during metadata unpacking
        print(f"Error: Failed to unpack metadata from {input_bw}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # General error catching for unexpected issues
        print(f"An error occurred during decoding {input_bw}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()