import pathlib
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.signal import find_peaks


def set_address(filename):
    """Set the address of the .mat file"""
    # Get the path to the .mat file from the current script location
    script_folder = pathlib.Path(__file__).parent.resolve()
    data_dir = pjoin(script_folder, "../", "data")
    mat_fname = pjoin(data_dir, filename)

    return mat_fname


def load_waveform_from_mat(filename):
    """
    Load a waveform from a .mat file

    Args:
        filename: Name of the .mat file (e.g., 'ecg.mat')

    Returns:
        t: time array
        signal: signal amplitude array
    """
    # Get the path to the .mat file
    script_folder = pathlib.Path(__file__).parent.resolve()
    data_dir = pjoin(script_folder, "../data")
    mat_fname = pjoin(data_dir, filename)

    # Load data
    var1 = "time"
    var2 = "ecg"
    data = sio.loadmat(mat_fname, variable_names=[var1, var2])

    return data[var1].squeeze(), data[var2].squeeze()


def count_peaks(signal, height=None, distance=None, prominence=None):
    """
    Count the number of peaks in a signal

    Args:
        signal: 1D array of signal values
        height: Minimum height of peaks (optional)
        distance: Minimum distance between peaks (optional)
        prominence: Minimum prominence of peaks (optional)

    Returns:
        num_peaks: Number of peaks detected
        peak_indices: Indices of the peaks
    """
    # Find peaks in the signal
    peak_indices, properties = find_peaks(
        signal, height=height, distance=distance, prominence=prominence
    )

    num_peaks = len(peak_indices)

    return num_peaks, peak_indices


def display_results(t, signal, num_peaks, peak_indices):
    """
    Display the waveform with detected peaks

    Args:
        t: time array
        signal: signal array
        num_peaks: number of peaks detected
        peak_indices: indices of peaks
    """
    print(f"\n{'=' * 50}")
    print(f"Peak Detection Results")
    print(f"{'=' * 50}")
    print(f"Number of peaks detected: {num_peaks}")
    print(f"Peak locations (indices): {peak_indices}")
    if len(t) == len(signal):
        print(f"Peak times: {t[peak_indices]}")
    print(f"Peak amplitudes: {signal[peak_indices]}")
    print(f"{'=' * 50}\n")

    # Plot the waveform with peaks marked
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, "b-", label="Signal")
    plt.plot(
        t[peak_indices],
        signal[peak_indices],
        "ro",
        markersize=8,
        label=f"Peaks (n={num_peaks})",
    )
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform with Detected Peaks (Total: {num_peaks})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    # Load waveform from .mat file
    # Try different files: 'ecg.mat', 'sin.mat', 'lab_e.mat'
    file = "ecg.mat"

    # Set address of data file
    filename = set_address(file)

    print(f"Loading waveform from {filename}...")
    t, signal = load_waveform_from_mat(filename)
    print(f"Loaded signal with {len(signal)} samples")

    # Count peaks
    # You can adjust these parameters for better peak detection:
    # - height: minimum peak height
    # - distance: minimum samples between peaks
    # - prominence: how much a peak stands out
    num_peaks, peak_indices = count_peaks(
        signal,
        height=None,  # Try: np.mean(signal)
        distance=20,  # Adjust based on your data
        prominence=1,
    )  # Adjust based on your data

    # Display results
    display_results(t, signal, num_peaks, peak_indices)


if __name__ == "__main__":
    main()
