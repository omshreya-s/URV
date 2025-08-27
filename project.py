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
from sklearn.model_selection import train_test_split, LeaveOneOut
from scipy.interpolate import interp1d
import neurokit2 as nk
from pathlib import Path

# Filter function (unchanged)
def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

# Upsampling function (to fs_target = 15 Hz)
def upsample_signal(time_orig, signal_orig, fs_new=15):

    duration = time_orig[-1] - time_orig[0]
    n_samples_new = int(duration * fs_new) + 1

    time_new = np.linspace(time_orig[0], time_orig[-1], n_samples_new)

    # Interpolator
    interpolator = interp1d(time_orig, signal_orig, kind='linear', fill_value="extrapolate")

    signal_new = interpolator(time_new)

    return time_new, signal_new

# EDA processing at native 15 Hz (original code, no upsampling):
def ea_detection(csv_file_path, fs=15):
    df = pd.read_csv(csv_file_path)
    df['time[s]'] = (df['LocalTimestamp'] - df['LocalTimestamp'].iloc[0])
    df = df.loc[(df['time[s]'] > 120) & (df['time[s]'] < (df['time[s]'].iloc[-1]) - 120)]
    ea_raw = df['EA'].astype(float)

    # Bandpass filter at original sampling rate
    ea_filtered = bandpass_filter(ea_raw, 0.1, 5, fs)
    time_ea = df['time[s]'].values

    return time_ea, ea_filtered

# bi features
def bi_features_from_window(bi_window: pd.Series):
    bi_window = bi_window.reset_index(drop=True)
    if len(bi_window) < 2:
        return {key: 0 for key in [
            'mean_bi', 'median_bi', 'std_bi', 'min_bi', 'max_bi',
            'minRatio_bi', 'maxRatio_bi', 'iqr_bi'
        ]}

    first_diff = bi_window.diff().dropna()
    second_diff = first_diff.diff().dropna()

    def safe_ratio(a, b):
        try:
            return float(a) / float(b) if float(b) != 0 else 0.0
        except Exception:
            return 0.0

    features = {
        'mean_bi': bi_window.mean(),
        'median_bi': bi_window.median(),
        'std_bi': bi_window.std(),
        'min_bi': bi_window.min(),
        'max_bi': bi_window.max(),
        'minRatio_bi': safe_ratio(bi_window.min(), bi_window.max()),
        'maxRatio_bi': safe_ratio(bi_window.max(), bi_window.min()),
        'iqr_bi': bi_window.quantile(0.75) - bi_window.quantile(0.25)

    }
    return features

def temp_features_from_window(temp_window):
    temp_window = temp_window.reset_index(drop=True)

    if len(temp_window) < 2:
        return {key: 0 for key in [
            'mean_temp', 'median_temp', 'std_temp', 'min_temp', 'max_temp',
            'minRatio_temp', 'maxRatio_temp', 'range_temp', 'iqr_temp', 'slope_temp'
        ]}

    def safe_ratio(a, b):
        return a / b if b != 0 else 0

    # deleted the first and second derivative here
    slope = np.polyfit(range(len(temp_window)), temp_window, 1)[0]

    features = {
        'mean_temp': temp_window.mean(),
        'median_temp': temp_window.median(),
        'std_temp': temp_window.std(),
        'min_temp': temp_window.min(),
        'max_temp': temp_window.max(),
        'minRatio_temp': safe_ratio(temp_window.min(), temp_window.max()),
        'maxRatio_temp': safe_ratio(temp_window.max(), temp_window.min()),
        'range_temp': temp_window.max() - temp_window.min(),
        'iqr_temp': temp_window.quantile(0.75) - temp_window.quantile(0.25),
        'slope_temp': slope
    }

    return features

# Windowed feature extraction updated to fs=15 (EDA native)
def windowed_feature_extraction(time_ea, ea_signal, time_hr, hr_signal, time_temp, temp_signal, time_bi, bi_signal, window_sec=5, fs=15, overlap=0.5, label='unknown'):
    window_size = int(window_sec * fs)  # samples per window
    step_size = int(window_size * (1 - overlap))  # step size between windows
    n_samples = len(ea_signal)

    features_list = []
    
    # Create HR dataframe for easy masking
    df_hr = pd.DataFrame({'time[s]': time_hr, 'HR': hr_signal})

    # Create T1 dataframe
    df_temp = pd.DataFrame({'time[s]': time_temp, 'T1': temp_signal})

    # Create BI dataframe
    df_bi = pd.DataFrame({'time[s]': time_bi, 'BI': bi_signal})

    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size

        ea_win = ea_signal[start:end]
        time_win = time_ea[start:end]

        # Process EDA window and extract EDA features
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

        # Get corresponding HR values for same time window (tolerance +/- small epsilon)
        hr_window = df_hr[(df_hr['time[s]'] >= time_win[0]) & (df_hr['time[s]'] <= time_win[-1])]['HR']

        # Skip windows with insufficient HR data
        if len(hr_window) < window_size * 0.9:  # 90% coverage threshold
            continue

        hr_feats = hr_features_from_window(hr_window)

        # Add temperature features
        temp_window = df_temp[(df_temp['time[s]'] >= time_win[0]) & (df_temp['time[s]'] <= time_win[-1])]['T1']
        temp_feats = temp_features_from_window(temp_window)

        # Add BI features
        bi_window = df_bi[(df_bi['time[s]'] >= time_win[0]) & (df_bi['time[s]'] <= time_win[-1])]['BI']
        bi_feats = bi_features_from_window(bi_window)

        combined_features = {**eda_features, **hr_feats, **temp_feats, **bi_feats, 'label': label}

        features_list.append(combined_features)

    return pd.DataFrame(features_list)

# HR features from window (same as before)
def hr_features_from_window(hr_window):
    hr_window = hr_window.reset_index(drop=True)
    if len(hr_window) < 2:
        # Not enough samples, return zeros
        return {key: 0 for key in [
            'mean_hr', 'median_hr', 'std_hr', 'min_hr', 'max_hr',
            'minRatio_hr', 'maxRatio_hr', 'median_first_derivative_hr',
            'min_first_derivative_hr', 'max_first_derivative_hr',
            'minRatio_first_derivative_hr', 'maxRatio_first_derivative_hr',
            'std_first_derivative_hr', 'min_second_derivative_hr',
            'max_second_derivative_hr', 'std_second_derivative_hr',
            'minRatio_second_derivative_hr', 'maxRatio_second_derivative_hr'
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
        'median_first_derivative_hr': first_diff.median(),
        'min_first_derivative_hr': first_diff.min(),
        'max_first_derivative_hr': first_diff.max(),
        'minRatio_first_derivative_hr': safe_ratio(first_diff.min(), first_diff.max()),
        'maxRatio_first_derivative_hr': safe_ratio(first_diff.max(), first_diff.min()),
        'std_first_derivative_hr': first_diff.std(),
        'min_second_derivative_hr': second_diff.min(),
        'max_second_derivative_hr': second_diff.max(),
        'std_second_derivative_hr': second_diff.std(),
        'minRatio_second_derivative_hr': safe_ratio(second_diff.min(), second_diff.max()),
        'maxRatio_second_derivative_hr': safe_ratio(second_diff.max(), second_diff.min())
    }
    return features

def pred_tree(frames):
    X = frames.drop('label', axis=1)  # drops 'label' column
    y = frames['label']

    def compute_subjectwise_norm_params(X_df, subj_series):
        params = {}
        for s in np.unique(subj_series):
            idx = (subj_series == s)
            mu = X_df.loc[idx].mean(axis=0)
            sd = X_df.loc[idx].std(axis=0, ddof=0)
            sd[sd == 0] = 1.0  # Avoid division by zero
            params[s] = (mu, sd)
        return params

    def apply_subjectwise_zscore(X_df, subj_series, norm_params):
        Xc = X_df.copy()
        for s in np.unique(subj_series):
            idx = (subj_series == s)
            if s in norm_params:
                mu, sd = norm_params[s]
                Xc.loc[idx, :] = (Xc.loc[idx, :] - mu) / sd
            else:
                # Subject not seen in training, fallback: optionally use global mean/std or leave as is
                # Here, we leave data unchanged; you can choose a better fallback as needed
                pass
        return Xc

    # Split data into train and test (random stratified split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Ensure 'subject_id' column exists
    if 'subject_id' not in X_train.columns:
        raise ValueError("Lack subject_id column")

    # Extract subject IDs for train and test
    subj_train = X_train['subject_id']
    subj_test = X_test['subject_id']

    # Drop 'subject_id' from feature sets
    X_train_feat = X_train.drop(columns=['subject_id'])
    X_test_feat = X_test.drop(columns=['subject_id'])

    # Compute normalization parameters on training data only
    norm_params = compute_subjectwise_norm_params(X_train_feat, subj_train)

    # Apply normalization to train and test sets using training params
    X_train_norm = apply_subjectwise_zscore(X_train_feat, subj_train, norm_params)
    X_test_norm = apply_subjectwise_zscore(X_test_feat, subj_test, norm_params)

    # Train Decision Tree Classifier
    dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    dt_model.fit(X_train_norm, y_train)
    y_pred = dt_model.predict(X_test_norm)
    acc = accuracy_score(y_test, y_pred)

    print(classification_report(y_test, y_pred))
    print("Decision Tree Accuracy:", acc)
    print("--------------------------------------------")

    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf_model.fit(X_train_norm, y_train)
    rf_pred = rf_model.predict(X_test_norm)

    print(classification_report(y_test, rf_pred))

    # Confusion matrix and plot
    rf_cm = confusion_matrix(y_test, rf_pred, labels=rf_model.classes_)
    disp2 = ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, display_labels=rf_model.classes_, cmap="BuGn")
    disp2.plot()

    print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

    print(X_train_norm.columns)

    # Feature importance from Random Forest
    importances = rf_model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': X_train_norm.columns, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False)
    print(feature_imp_df)

    plt.figure(figsize=(8, 4))
    topk = min(10, len(feature_imp_df))
    plt.barh(feature_imp_df.iloc[:topk]['Feature'], feature_imp_df.iloc[:topk]['Gini Importance'], color='lightblue')
    plt.xlabel('Gini Importance')
    plt.title('Feature Importance - Gini Importance')
    plt.gca().invert_yaxis()
    plt.show()

def pred_tree_LOSO(frames):
    if 'subject_id' not in frames.columns:
        raise ValueError("Lack subject_id column")

    subjects = frames['subject_id'].unique()
    
    acc_results_dt = []
    acc_results_rf = []
    all_y_true = []
    all_y_pred_dt = []
    all_y_pred_rf = []
    
    # Initialize feature importance accumulator for RF
    feature_importance_accum = None
    feature_names = None
    
    def compute_subjectwise_norm_params(X_df, subj_series):
        params = {}
        for s in np.unique(subj_series):
            idx = (subj_series == s)
            mu = X_df.loc[idx].mean(axis=0)
            sd = X_df.loc[idx].std(axis=0, ddof=0)
            sd[sd==0] = 1.0
            params[s] = (mu, sd)
        return params

    def apply_subjectwise_zscore(X_df, subj_series, norm_params):
        Xc = X_df.copy()
        for s in np.unique(subj_series):
            idx = (subj_series == s)
            if s in norm_params:
                mu, sd = norm_params[s]
                Xc.loc[idx, :] = (Xc.loc[idx, :] - mu) / sd
            else:
                # If subject not in training norm params, leave as is or apply global mean/std if desired
                pass
        return Xc

    for test_subj in subjects:
        # Split data into train (all other subjects) and test (left-out subject)
        train_data = frames[frames['subject_id'] != test_subj]
        test_data = frames[frames['subject_id'] == test_subj]

        X_train = train_data.drop(columns=['label'])
        y_train = train_data['label']
        subj_train = X_train['subject_id']
        X_train_feat = X_train.drop(columns=['subject_id'])

        X_test = test_data.drop(columns=['label'])
        y_test = test_data['label']
        subj_test = X_test['subject_id']
        X_test_feat = X_test.drop(columns=['subject_id'])

        # Compute normalization params on training data
        norm_params = compute_subjectwise_norm_params(X_train_feat, subj_train)

        # Normalize training and test data using training normalization params
        X_train_norm = apply_subjectwise_zscore(X_train_feat, subj_train, norm_params)
        X_test_norm = apply_subjectwise_zscore(X_test_feat, subj_test, norm_params)

        # Train Decision Tree
        dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
        dt_model.fit(X_train_norm, y_train)
        y_pred_dt = dt_model.predict(X_test_norm)
        acc_dt = accuracy_score(y_test, y_pred_dt)
        acc_results_dt.append(acc_dt)

        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        rf_model.fit(X_train_norm, y_train)
        y_pred_rf = rf_model.predict(X_test_norm)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        acc_results_rf.append(acc_rf)

        # Aggregate ground truth and predictions for overall metrics later
        all_y_true.extend(y_test)
        all_y_pred_dt.extend(y_pred_dt)
        all_y_pred_rf.extend(y_pred_rf)

        # Accumulate feature importances for RF
        if feature_importance_accum is None:
            feature_importance_accum = rf_model.feature_importances_
            feature_names = X_train_norm.columns
        else:
            feature_importance_accum += rf_model.feature_importances_

        print(f"Subject {test_subj} left out:")
        print(f"  Decision Tree Accuracy: {acc_dt:.4f}")
        print(f"  Random Forest Accuracy: {acc_rf:.4f}")
        print("-------------------------------")

    # Overall performance
    print("===== Overall Decision Tree Classification Report =====")
    print(classification_report(all_y_true, all_y_pred_dt))
    print("===== Overall Random Forest Classification Report =====")
    print(classification_report(all_y_true, all_y_pred_rf))

    print(f"Mean Decision Tree Accuracy (LOSO): {np.mean(acc_results_dt):.4f}")
    print(f"Mean Random Forest Accuracy (LOSO): {np.mean(acc_results_rf):.4f}")

    # Average feature importance over folds
    avg_feature_importance = feature_importance_accum / len(subjects)
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Avg Gini Importance': avg_feature_importance}).sort_values('Avg Gini Importance', ascending=False)
    print(feature_imp_df)

    plt.figure(figsize=(8, 4))
    topk = min(10, len(feature_imp_df))
    plt.barh(feature_imp_df.iloc[:topk]['Feature'], feature_imp_df.iloc[:topk]['Avg Gini Importance'], color='lightblue')
    plt.xlabel('Average Gini Importance')
    plt.title('Feature Importance - Average Gini Importance (LOSO)')
    plt.gca().invert_yaxis()
    plt.show()

    # Confusion matrix for Random Forest overall predictions
    disp = ConfusionMatrixDisplay.from_predictions(all_y_true, all_y_pred_rf, cmap="BuGn")
    disp.plot()
    plt.title("Random Forest Confusion Matrix - Overall (LOSO)")
    plt.show()

def process_feature(state, feature, x, y, fs_eda):
    feature_filename = state + "_" + feature + "\\" + y[1] + "_" + x[2] + "_" + feature +".csv"
    df = pd.read_csv(feature_filename)
    print("processing " + feature_filename)

    # trim
    df['time[s]'] = (df['LocalTimestamp'] - df['LocalTimestamp'].iloc[0])
    df = df.loc[(df['time[s]'] > 120) & (df['time[s]'] < (df['time[s]'].iloc[-1]) - 120)]

    # Upsample to align with EDA
    time_up, signal_up = upsample_signal(df['time[s]'].values, df[feature].values, fs_new=fs_eda)

    return time_up, signal_up

def read_files(files, label, data_frame, fs_eda, window_sec, overlap):
    for file in files:
        subj = Path(file).stem  
        time_ea, ea_filtered = ea_detection(file, fs=fs_eda)


        x = file.split("_")
        y = x[1].split("\\")

        # get time and value signals after processing
        time_hr_up, hr_up = process_feature(label, "HR", x, y, fs_eda)
        time_temp_up, temp_up = process_feature(label, "T1", x, y, fs_eda)
        time_bi_up, bi_up = process_feature(label, "BI", x, y, fs_eda)

        windowed_df = windowed_feature_extraction(time_ea, ea_filtered, time_hr_up, hr_up, time_temp_up, temp_up, time_bi_up, bi_up,
                                                    window_sec=window_sec, fs=fs_eda,
                                                    overlap=overlap, label=label)
        
        print(f"{label} file {file} generated {len(windowed_df)} windowed samples")

        if not windowed_df.empty:
            windowed_df["subject_id"] = subj
            data_frame = pd.concat([data_frame, windowed_df], ignore_index=True)
            print(f"{len(data_frame)}")
        else:
            print(f"Warning: No valid windows extracted from {label} file {file}")
    return data_frame

# replace the "\\" with "/" if file not found
def main():
    fs_eda = 15
    window_sec = 10
    overlap = 0.5

    data = pd.DataFrame()

    # Engaged files
    engaged_filenames = glob.glob("engaged_EA/*.csv")
    data = read_files(engaged_filenames, "engaged", data, fs_eda, window_sec, overlap)
    print(f"Total samples: {len(data)}")
    
    # Relaxed files
    relaxed_filenames = glob.glob("relaxed_EA/*.csv")
    data = read_files(relaxed_filenames, "relaxed", data, fs_eda, window_sec, overlap)

    print(f"Total samples: {len(data)}")
    pred_tree(data)

if __name__ == '__main__':
    main()