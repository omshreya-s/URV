import glob
import pandas as pd
import os
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
import neurokit2 as nk


# filters
def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    y = filtfilt(b, a, data)
    return y

def bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)


# EDA features
def ea_detection(csv_file_path):
    # Load data
    df = pd.read_csv(csv_file_path)
    df['time[s]'] = (df['LocalTimestamp'] - df['LocalTimestamp'].iloc[0])

    df = df.loc[(df['time[s]'] > 120) & (df['time[s]']  < (df['time[s]'].iloc[-1])-120)]

    ea_raw = df['EA'].astype(float)
  
    time = df['time[s]']
    sampling_rate = 15          
    
    ea_filtered = bandpass_filter(ea_raw, 0.1, 5, sampling_rate)

    df['filtered_EA'] = ea_filtered


    signals, info = nk.eda_process(ea_filtered, sampling_rate=15, method='neurokit', report=None)
  

    return signals[['SCR_Onsets', 'SCR_Peaks',  'SCR_Height' , 'SCR_Amplitude', 'SCR_RiseTime', 'SCR_Recovery', 'SCR_RecoveryTime']]

# HR features

def hr_detection(csv_file_path):
    
    df = pd.read_csv(csv_file_path)
    df['time[s]'] = (df['LocalTimestamp'] - df['LocalTimestamp'].iloc[0])

    df = df.loc[(df['time[s]'] > 120) & (df['time[s]']  < (df['time[s]'].iloc[-1])-120)]
    features = {
        'mean': df['HR'].mean(),
        'median': df['HR'].median(),
        'std': df['HR'].std(),
        'min': df['HR'].min(),
        'max': df['HR'].max(),
        'minRatio': df['HR'].min() / df['HR'].max(),
        'maxRatio': df['HR'].max() / df['HR'].min(),
        # median of the first derivative
        'median_first_derivative': df['HR'].diff().median(),
        # min and max of the first derivative
        'min_first_derivative': df['HR'].diff().min(),
        'max_first_derivative': df['HR'].diff().max(),
        # min ratio and max ratio of the first derivative
        'minRatio_first_derivative': df['HR'].diff().min() / df['HR'].diff().max(),
        'maxRatio_first_derivative': df['HR'].diff().max() / df['HR'].diff().min(),
        # standard deviation of the first derivative
        'std_first_derivative': df['HR'].diff().std(),
        # min and max of the second derivative
        'min_second_derivative': df['HR'].diff().diff().min(),
        'max_second_derivative': df['HR'].diff().diff().max(),
        # standard deviation of the second derivative
        'std_second_derivative': df['HR'].diff().diff().std(),
        # min ratio and max ratio of the second derivative
        'minRatio_second_derivative': df['HR'].diff().diff().min() / df['HR'].diff().diff().max(),
        'maxRatio_second_derivative': df['HR'].diff().diff().max() / df['HR'].diff().diff().min()
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

# Run detection

engaged_filenames = glob.glob("engaged_EA/*.csv")

data = pd.DataFrame()

for file in engaged_filenames:
    # raw_data = pd.read_csv(file)
    # graph(raw_data)
    df = ea_detection(file)

    #creating HR filename to extract HR features
    x = file.split("_")
    y = x[1].split("\\")

    hr_name ="engaged_HR" + "\\" + y[1] +"_" + x[2]+ "_HR.csv"

    #extracting hr features
    hr_df = pd.DataFrame(hr_detection(hr_name), index = [0])

    for (columnName, columnData) in hr_df.items():
        df[columnName] = columnData[0]

    df['label'] =  'engaged'

    data = pd.concat([data, df], ignore_index = True)
  

relaxed_filenames = glob.glob("relaxed_EA/*.csv")

for file in relaxed_filenames:
    # raw_data = pd.read_csv(file)
    # graph(raw_data)
    df = ea_detection(file)

    #creating HR filename to extract HR features
    x = file.split("_")
    y = x[1].split("\\")

    hr_name ="relaxed_HR" + "\\" + y[1] +"_" + x[2]+ "_HR.csv"
    
    #extracting hr features
    hr_df = pd.DataFrame(hr_detection(hr_name), index = [0])

    for (columnName, columnData) in hr_df.items():
        df[columnName] = columnData[0]

    df['label'] = 'relaxed'

    data = pd.concat([data, df], ignore_index = True)


pred_tree(data)