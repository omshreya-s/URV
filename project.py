import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np
from sklearn import tree
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import neurokit2 as nk

# Filter function (unchanged)
def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

# HR upsampling function (to fs_target = 15 Hz)
def upsample_hr_signal(hr_time_orig, hr_signal_orig, fs_target=15):
    duration = hr_time_orig[-1] - hr_time_orig[0]
    n_samples_new = int(duration * fs_target) + 1
    time_new = np.linspace(hr_time_orig[0], hr_time_orig[-1], n_samples_new)
    interpolator = interp1d(hr_time_orig, hr_signal_orig, kind='linear', fill_value='extrapolate')
    signal_new = interpolator(time_new)
    return time_new, signal_new

# EDA processing at native 15 Hz (original code, no upsampling):
def ea_detection(csv_file_path, fs=15):
    df = pd.read_csv(csv_file_path)
    df['time[s]'] = (df['LocalTimestamp'] - df['LocalTimestamp'].iloc[0])
    df = df.loc[(df['time[s]'] > 120) & (df['time[s]'] < (df['time[s]'].iloc[-1]) - 120)]

    ea_raw = df['EA'].astype(float)

    ea_filtered = bandpass_filter(ea_raw, 0.1, 5, fs)
    time_ea = df['time[s]'].values
    return time_ea, ea_filtered

# Windowed feature extraction updated to fs=15 (EDA native)
def windowed_feature_extraction(time_ea, ea_signal, time_hr, hr_signal, window_sec=5, fs=15, overlap=0.5, label='unknown'):
    window_size = int(window_sec * fs)
    step_size = int(window_size * (1 - overlap))
    n_samples = len(ea_signal)
    features_list = []
    
    # Create HR dataframe for easy masking
    df_hr = pd.DataFrame({'time[s]': time_hr, 'HR': hr_signal})
    
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        ea_win = ea_signal[start:end]
        time_win = time_ea[start:end]
        
        # Process EDA window
        signals, _ = nk.eda_process(ea_win, sampling_rate=fs, method='neurokit')
        eda_features = {
            'SCR_Onsets_sum': signals['SCR_Onsets'].sum(),
            'SCR_Peaks_sum': signals['SCR_Peaks'].sum(),
            'SCR_Height_mean': signals['SCR_Height'].mean() if len(signals) > 0 else 0,
            'SCR_Amplitude_mean': signals['SCR_Amplitude'].mean() if len(signals) > 0 else 0,
            'SCR_RiseTime_mean': signals['SCR_RiseTime'].mean() if len(signals) > 0 else 0,
            'SCR_Recovery_mean': signals['SCR_Recovery'].mean() if len(signals) > 0 else 0,
            'SCR_RecoveryTime_mean': signals['SCR_RecoveryTime'].mean() if len(signals) > 0 else 0,
        }
        
        # Select HR window with tolerance
        eps = 0.02  # 20 ms tolerance
        hr_win = df_hr[(df_hr['time[s]'] >= time_win[0] - eps) & (df_hr['time[s]'] <= time_win[-1] + eps)]['HR']
        
        if len(hr_win) < window_size * 0.5:  # relaxed requirement for coverage
            continue
        
        # HR features per window (same as before)
        hr_feats = hr_features_from_window(hr_win)
        
        combined_features = {**eda_features, **hr_feats, 'label': label}
        features_list.append(combined_features)
    
    return pd.DataFrame(features_list)

# HR features from window (same as before)
def hr_features_from_window(hr_window):
    hr_window = hr_window.reset_index(drop=True)
    if len(hr_window) < 2:
        return {key: 0 for key in [
            'mean_hr', 'median_hr', 'std_hr', 'min_hr', 'max_hr',
            'minRatio_hr', 'maxRatio_hr', 'median_first_derivative',
            'min_first_derivative', 'max_first_derivative',
            'minRatio_first_derivative', 'maxRatio_first_derivative',
            'std_first_derivative', 'min_second_derivative',
            'max_second_derivative', 'std_second_derivative',
            'minRatio_second_derivative', 'maxRatio_second_derivative'
        ]}
    first_diff = hr_window.diff().dropna()
    second_diff = first_diff.diff().dropna()
    def safe_ratio(a, b):
        return a / b if b != 0 else 0
    features = {
        'mean_hr': hr_window.mean(),
        'median_hr': hr_window.median(),
        'std_hr': hr_window.std(),
        'min_hr': hr_window.min(),
        'max_hr': hr_window.max(),
        'minRatio_hr': safe_ratio(hr_window.min(), hr_window.max()),
        'maxRatio_hr': safe_ratio(hr_window.max(), hr_window.min()),
        'median_first_derivative': first_diff.median(),
        'min_first_derivative': first_diff.min(),
        'max_first_derivative': first_diff.max(),
        'minRatio_first_derivative': safe_ratio(first_diff.min(), first_diff.max()),
        'maxRatio_first_derivative': safe_ratio(first_diff.max(), first_diff.min()),
        'std_first_derivative': first_diff.std(),
        'min_second_derivative': second_diff.min(),
        'max_second_derivative': second_diff.max(),
        'std_second_derivative': second_diff.std(),
        'minRatio_second_derivative': safe_ratio(second_diff.min(), second_diff.max()),
        'maxRatio_second_derivative': safe_ratio(second_diff.max(), second_diff.min())
    }
    return features


def pred_tree(frames):
    X = frames.drop('label', axis=1) #drops 'label' column
    y = frames['label']

    #splits into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=3)

    #trains model?
    dt_model.fit(X_train, y_train)

    #predicts 'y' values with test 'x' values
    y_pred = dt_model.predict(X_test)

    #checks accuracy of predicted 'y' against true 'y'
    acc = accuracy_score(y_test, y_pred)

    # confusion matrix
    dt_cm = confusion_matrix(y_test, y_pred, labels=dt_model.classes_)

    # precision, recall, f1 score
    print(classification_report(y_test, y_pred))

    print("Decision Tree Accuracy:", acc)

    print("--------------------------------------------")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    print(classification_report(y_test, rf_pred))
    print("Random Forest Accuracy:", rf_model.score(X_test, y_test))


def main():
    fs_eda = 15  # EDA original fs
    window_sec = 5
    overlap = 0.5

    data = pd.DataFrame()
    engaged_files = glob.glob("engaged_EA/*.csv")
    for file in engaged_files:
        time_ea, ea_filtered = ea_detection(file, fs=fs_eda)
        
        # Prepare HR filename and load HR
        parts = file.split("_")
        yparts = parts[1].split("\\")
        hr_name = "engaged_HR\\" + yparts[1] + "_" + parts[2] + "_HR.csv"
        print(hr_name)
        df_hr = pd.read_csv(hr_name)
        df_hr['time[s]'] = (df_hr['LocalTimestamp'] - df_hr['LocalTimestamp'].iloc[0])
        
        # Upsample HR to 15 Hz to align with EDA
        time_hr_up, hr_up = upsample_hr_signal(df_hr['time[s]'].values, df_hr['HR'].values, fs_target=fs_eda)
        
        # Extract features with windowing
        windowed_df = windowed_feature_extraction(time_ea, ea_filtered, time_hr_up, hr_up,
                                                  window_sec=window_sec, fs=fs_eda,
                                                  overlap=overlap, label='engaged')
        if not windowed_df.empty:
            data = pd.concat([data, windowed_df], ignore_index=True)


    relaxed_files = glob.glob("relaxed_EA/*.csv")
    for file in relaxed_files:
        time_ea, ea_filtered = ea_detection(file, fs=fs_eda)
        
        # Prepare HR filename and load HR
        parts = file.split("_")
        yparts = parts[1].split("\\")
        hr_name = "relaxed_HR\\" + yparts[1] + "_" + parts[2] + "_HR.csv"
        print(hr_name)
        df_hr = pd.read_csv(hr_name)
        df_hr['time[s]'] = (df_hr['LocalTimestamp'] - df_hr['LocalTimestamp'].iloc[0])
        
        # Upsample HR to 15 Hz to align with EDA
        time_hr_up, hr_up = upsample_hr_signal(df_hr['time[s]'].values, df_hr['HR'].values, fs_target=fs_eda)
        
        # Extract features with windowing
        windowed_df = windowed_feature_extraction(time_ea, ea_filtered, time_hr_up, hr_up,
                                                  window_sec=window_sec, fs=fs_eda,
                                                  overlap=overlap, label='relaxed')
        if not windowed_df.empty:
            data = pd.concat([data, windowed_df], ignore_index=True)

    print(f"Total samples: {len(data)}")
    pred_tree(data)

if __name__ == '__main__':
    main()