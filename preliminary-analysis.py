import numpy as np
from scipy.io import wavfile
import warnings

# Suppress harmless WavFileWarning
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

filepath = 'data/7008d9c8-6868-47eb-9935-3cf6885cdb1d.wav'

try:
    sample_rate, data = wavfile.read(filepath)

    print(f"Successfully loaded: {filepath}")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Data Type: {data.dtype}")
    print(f"Number of Samples: {len(data)}")
    print(f"Duration: {len(data) / sample_rate:.2f} seconds")
    print(f"Min Value: {np.min(data)}")
    print(f"Max Value: {np.max(data)}")
    print(f"Mean Value: {np.mean(data):.2f}")
    print(f"Standard Deviation: {np.std(data):.2f}")
    print("\nFirst 10 samples:")
    print(data[:10])
    print("\nLast 10 samples:")
    print(data[-10:])

except FileNotFoundError:
    print(f"Error: File not found at {filepath}")
except Exception as e:
    print(f"An error occurred: {e}")