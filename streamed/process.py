#!/usr/bin/env python3
import time
import os
import csv
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from bitalino import BITalino

# -------------------------------------------
# USER CONFIGURATION
# -------------------------------------------
SERIAL_PORT          = "/dev/tty.BITalino-39-F6"  # update to your port
SAMPLING_RATE        = 1000                             # Hz
CHANNELS             = [0, 1, 2, 3, 4, 5]
ECG_INDEX            = CHANNELS.index(5)               # channel 5 is ECG
EMG_INDEX            = CHANNELS.index(4)               # channel 4 is EMG
N_FRAMES             = 500                              # samples per read
RUN_DURATION         = 20                               # seconds per run
CSV_TEMPLATE         = "bitalino_run_{:02d}.csv"

# Pan–Tompkins ECG parameters
ECG_INT_WINDOW_SEC   = 0.080                            # 80 ms
ECG_INT_WINDOW_SAMPS = int(ECG_INT_WINDOW_SEC * SAMPLING_RATE)

# EMG analysis parameters
EMG_LOWCUT           = 100.0    # Hz
EMG_HIGHCUT          = 300.0    # Hz
EMG_FILTER_ORDER     = 3        # bandpass filter order
EMG_MOVING_AVG_SEC   = 0.1      # 100 ms window
EMG_MOVING_AVG_SAMPS = int(EMG_MOVING_AVG_SEC * SAMPLING_RATE)
EMG_RMS_THRESHOLD    = 31    # threshold for "tension"

# -------------------------------------------
# ECG Analysis (Pan–Tompkins)
# -------------------------------------------
# Pan–Tompkins bandpass design parameters
LOWCUT         = 5.0    # Hz
HIGHCUT        = 15.0   # Hz
FILTER_ORDER   = 2

def analyze_ecg(ecg_signal: np.ndarray, fs: int):
    """
    Pan–Tompkins pipeline on ecg_signal sampled at fs,
    printing beats, HR, mean RR, SDNN, and RMSSD.
    """
    # 1) Band-pass filter
    nyq = 0.5 * fs
    low = LOWCUT / nyq
    high = HIGHCUT / nyq
    b, a = butter(2, [low, high], btype='band')
    filtered = filtfilt(b, a, ecg_signal)

    # 2) Derivative
    deriv = np.diff(filtered, prepend=filtered[0])

    # 3) Squaring
    squared = deriv**2

    # 4) Moving-average integration
    kernel = np.ones(ECG_INT_WINDOW_SAMPS) / ECG_INT_WINDOW_SAMPS
    integ = np.convolve(squared, kernel, mode='same')

    # 5) R-peak detection
    thresh = np.mean(integ)
    distance = int(0.2 * fs)
    peaks, _ = find_peaks(integ, height=thresh, distance=distance)

    # 6) HR & HRV metrics
    if len(peaks) >= 2:
        rr_secs = np.diff(peaks) / fs               # seconds
        hr_bpm = 60.0 / np.mean(rr_secs)
        rr_ms = rr_secs * 1000                      # ms
        mean_rr = np.mean(rr_ms)
        sdnn = np.std(rr_ms)
        # RMSSD: root mean square of successive RR diffs
        diff_ms = np.diff(rr_ms)
        rmssd = np.sqrt(np.mean(diff_ms**2)) if diff_ms.size>0 else 0.0
    else:
        hr_bpm = mean_rr = sdnn = rmssd = 0.0
    status = ""
    if rmssd < 20:
        status = "stressed"
    else:
        status = "notStressed"
    print("🎯 ECG Analysis Results:")
    print(f"  • detected beats    : {len(peaks)}")
    print(f"  • Heart Rate       : {hr_bpm:.1f} bpm")
    print(f"  • Mean RR interval : {mean_rr:.1f} ms")
    print(f"  • SDNN (HRV)       : {sdnn:.1f} ms")
    print(f"  • RMSSD (HRV)      : {rmssd:.1f} ms\n")
    return status

# -------------------------------------------
# EMG Analysis
# -------------------------------------------
def analyze_emg(emg_signal: np.ndarray, fs: int):
    """
    Processes emg_signal sampled at fs and prints RMS, iEMG,
    and a tension verdict based on RMS threshold.
    """
    # 1) Band-pass filter (100–300 Hz)
    nyq = 0.5 * fs
    low, high = EMG_LOWCUT/nyq, EMG_HIGHCUT/nyq
    b, a = butter(EMG_FILTER_ORDER, [low, high], btype='band')
    filtered = filtfilt(b, a, emg_signal)

    # 2) Rectify
    rectified = np.abs(filtered)

    # 3) Moving-average smoothing
    kernel = np.ones(EMG_MOVING_AVG_SAMPS) / EMG_MOVING_AVG_SAMPS
    mov_avg = np.convolve(rectified, kernel, mode='same')

    # 4) Compute RMS and iEMG
    rms = np.sqrt(np.mean(mov_avg**2))
    iemg = np.trapz(mov_avg, dx=1/fs)

    status = "TENSED" if rms > EMG_RMS_THRESHOLD else "RELAXED"

    print(f"🎯 EMG Analysis Results:")
    print(f"  • RMS value : {rms:.2f}")
    print(f"  • iEMG value: {iemg:.2f}")
    print(f"  • Status    : {status} (thr={EMG_RMS_THRESHOLD})")
    return status

# -------------------------------------------
# Capture & Analyze
# -------------------------------------------
def capture_run(device, run_idx: int):
    # Prepare CSV file
    filename = CSV_TEMPLATE.format(run_idx)
    if os.path.exists(filename): os.remove(filename)
    header = ["timestamp"] + [f"ch{i}" for i in CHANNELS]
    first = True

    # Buffers
    ecg_buf = []
    emg_buf = []

    print(f"\n🔴 Capturing 10 s run #{run_idx} → {filename}")
    device.start(SAMPLING_RATE, CHANNELS)
    t0 = time.time()
    while time.time() - t0 < RUN_DURATION:
        data = device.read(N_FRAMES)
        # buffer channels
        ecg_buf.extend(data[:, 1 + ECG_INDEX])
        emg_buf.extend(data[:, 1 + EMG_INDEX])
        # write CSV
        mode = 'w' if first else 'a'
        with open(filename, mode, newline='') as f:
            w = csv.writer(f)
            if first:
                w.writerow(header)
                first = False
            w.writerows(data.tolist())
        print(f"  • wrote {len(data)} samples; elapsed {time.time()-t0:.2f}s")
    device.stop()
    print(f"✅ Run #{run_idx} saved → {filename}")

    # Convert buffers to arrays
    ecg_arr = np.array(ecg_buf, dtype=float)
    emg_arr = np.array(emg_buf, dtype=float)

    # Analyze
    print(analyze_ecg(ecg_arr, SAMPLING_RATE))
    print(analyze_emg(emg_arr, SAMPLING_RATE))

# -------------------------------------------
# Main Loop
# -------------------------------------------
def main():
    try:
        dev = BITalino(SERIAL_PORT)
        print(f"✅ Connected to BITalino on {SERIAL_PORT}")
    except Exception as e:
        print("❌ Connection failed:", e)
        return
    run = 1
    print("Type 'start' to capture+analyze; 'exit' to quit.\n")
    while True:
        cmd = input(f"[run {run}] > ").strip().lower()
        if cmd == 'start':
            capture_run(dev, run)
            run += 1
        elif cmd in ('exit','quit'):
            break
        else:
            print("Unknown command. Use 'start' or 'exit'.")
    dev.close()
    print("👋 Goodbye!")

if __name__ == '__main__':
    main()
