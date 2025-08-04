import pandas as pd

# load EDA data
base_path = r"data\engaged task(normal sitting)  (1)\july10_day1 (1)\emotibit (1)\2025-07-10_15-09-52-005859"
ea_file = base_path + "_EA.csv"
sa_file = base_path + "_SA.csv"
sf_file = base_path + "_SF.csv"
sr_file = base_path + "_SR.csv"
hr_file = base_path + "_HR.csv"

# EDA
df_ea = pd.read_csv(ea_file)
# SCR Amplitude
df_sa = pd.read_csv(sa_file)
# SCR Frequency
df_sf = pd.read_csv(sf_file)
# SCR Rise Time
df_sr = pd.read_csv(sr_file)

# cut off the first and last 120 seconds
min_ea = df_ea['LocalTimestamp'].min() + 120
max_ea = df_ea['LocalTimestamp'].max() - 120
min_sa = df_sa['LocalTimestamp'].min() + 120
max_sa = df_sa['LocalTimestamp'].max() - 120
min_sf = df_sf['LocalTimestamp'].min() + 120
max_sf = df_sf['LocalTimestamp'].max() - 120
min_sr = df_sr['LocalTimestamp'].min() + 120
max_sr = df_sr['LocalTimestamp'].max() - 120


df_ea = df_ea[(df_ea['LocalTimestamp'] > min_ea) & (df_ea['LocalTimestamp'] < max_ea)]
df_sa = df_sa[(df_sa['LocalTimestamp'] > min_sa) & (df_sa['LocalTimestamp'] < max_sa)]
df_sr = df_sr[(df_sr['LocalTimestamp'] > min_sr) & (df_sr['LocalTimestamp'] < max_sr)]

# extract features: mean value, median value, standard deviation, min value, max value, minRatio, maxRatio and median value of first difference, min values, max values, standard deviation minRatio and maxRatio of first and second differences.
def extract_eda_features(df):
    features = {
        'mean': df['EA'].mean(),
        'median': df['EA'].median(),
        'std': df['EA'].std(),
        'min': df['EA'].min(),
        'max': df['EA'].max(),
        'minRatio': df['EA'].min() / df['EA'].max(),
        'maxRatio': df['EA'].max() / df['EA'].min(),
        # median of the first derivative
        'median_first_derivative': df['EA'].diff().median(),
        # min and max of the first derivative
        'min_first_derivative': df['EA'].diff().min(),
        'max_first_derivative': df['EA'].diff().max(),
        # min ratio and max ratio of the first derivative
        'minRatio_first_derivative': df['EA'].diff().min() / df['EA'].diff().max(),
        'maxRatio_first_derivative': df['EA'].diff().max() / df['EA'].diff().min(),
        # standard deviation of the first derivative
        'std_first_derivative': df['EA'].diff().std(),
        # min and max of the second derivative
        'min_second_derivative': df['EA'].diff().diff().min(),
        'max_second_derivative': df['EA'].diff().diff().max(),
        # standard deviation of the second derivative
        'std_second_derivative': df['EA'].diff().diff().std(),
        # min ratio and max ratio of the second derivative
        'minRatio_second_derivative': df['EA'].diff().diff().min() / df['EA'].diff().diff().max(),
        'maxRatio_second_derivative': df['EA'].diff().diff().max() / df['EA'].diff().diff().min()
    }
    return features
ea_features = extract_eda_features(df_ea)

# print EDA features
print("Extracted EDA Features:")
for key, value in ea_features.items():
    print(f"{key}: {value}")

# extract features for SCR amplitude:
def extract_sa_features(df):
    features = {
        'mean': df['SA'].mean(),
        'median': df['SA'].median(),
        'std': df['SA'].std(),
        'max': df['SA'].max(),
        'min': df['SA'].min()
    }
    return features
sa_features = extract_sa_features(df_sa)
print("\nExtracted SCR Amplitude Features:")
for key, value in sa_features.items():
    print(f"{key}: {value}")

def extract_sr_features(df):
    features = {
        'mean': df['SR'].mean(),
        'median': df['SR'].median(),
        'std': df['SR'].std(),
        'max': df['SR'].max(),
        'min': df['SR'].min()
    }
    return features
sr_features = extract_sr_features(df_sr)
print("\nExtracted SCR Rise Time Features:")
for key, value in sr_features.items():
    print(f"{key}: {value}")




# plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df_ea['LocalTimestamp'], df_ea['EA'], label='Original EDA', color='blue')
plt.plot(df_sa["LocalTimestamp"], df_sa['SA'], "ro", label='SCR Amplitude')
# plt.plot(df_sf["LocalTimestamp"], df_sf['SF'], "go", label='SCR Frequency')
plt.plot(df_sr["LocalTimestamp"], df_sr['SR'], "yo", label='SCR Rise Time')
plt.xlabel('Local Timestamp')
plt.ylabel('EDA')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()