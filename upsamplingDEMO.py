import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def upsample_signal(time_orig, signal_orig, fs_new=15):
    duration = time_orig[-1] - time_orig[0]
    n_samples_new = int(duration * fs_new) + 1
    time_new = np.linspace(time_orig[0], time_orig[-1], n_samples_new)
    interpolator = interp1d(time_orig, signal_orig, kind='linear', fill_value="extrapolate")
    signal_new = interpolator(time_new)
    return time_new, signal_new

# Replace this path with one of your actual HR file paths
hr_csv_file = r"engaged_HR\2025-07-10_15-09-52-005859_HR.csv"

# Load HR data
df_hr = pd.read_csv(hr_csv_file)

# Compute relative time in seconds
df_hr['time[s]'] = (df_hr['LocalTimestamp'] - df_hr['LocalTimestamp'].iloc[0])
df_hr = df_hr.loc[(df_hr['time[s]'] > 120) & (df_hr['time[s]'] < (df_hr['time[s]'].iloc[-1]) - 450)]
print(f"OG Samples: {len(df_hr)}")
# Original HR timestamps and signal
time_orig = df_hr['time[s]'].values
hr_orig = df_hr['HR'].values

# Upsample to 15 Hz
fs_new = 15
time_up, hr_up = upsample_signal(time_orig, hr_orig, fs_new)
print(f"New samples: {len(hr_up)}")

# Plot original HR and upsampled HR
plt.figure(figsize=(12, 5))
plt.plot(time_orig, hr_orig, 'ro-', label='Original HR (sparse samples) at around 1.5 Hz')
plt.plot(time_up, hr_up, '.-', label=f'Upsampled HR at {fs_new} Hz', alpha=0.5, color="#3e809c")
plt.xlabel('Time [s]')
plt.ylabel('HR')
plt.title('HR signal before and after upsampling')
plt.legend()
plt.grid(True)
plt.show()