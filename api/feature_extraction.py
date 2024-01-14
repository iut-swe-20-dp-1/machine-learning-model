import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler

def statistical_features(arr):
    vmin = np.amin(arr)
    vmax = np.amax(arr)
    mean = np.mean(arr)
    std = np.std(arr)
    return vmin, vmax, mean, std

def shape_features(arr):
    skewness = skew(arr)
    kurt = kurtosis(arr)
    return skewness, kurt

def calculate_rms(signal):
    diff_squared = np.square(np.ediff1d(signal))
    rms_value = np.sqrt(np.mean(diff_squared))
    return rms_value

def extract_features(data):
    cols = [
        'EDA_Mean', 'EDA_Min', 'EDA_Max', 'EDA_Std', 'EDA_Kurtosis', 'EDA_Skew', 'EDA_Num_Peaks', 'EDA_Amplitude', 'EDA_Duration',
        'HR_Mean', 'HR_Min', 'HR_Max', 'HR_Std', 'HR_RMS', 'TEMP_Mean', 'TEMP_Min', 'TEMP_Max', 'TEMP_Std'
    ]

    df_features = pd.DataFrame(columns=cols)
    index = 0

    for i in tqdm(range(0, len(data['EDA']), 20), desc="Processing rows", leave=True):
        df_partial = data.iloc[i:i+40,]
        plen = len(df_partial['EDA'])

        if plen < 40:
            continue

        eda = df_partial['EDA'].values
        hr = df_partial['HR'].values
        temp = df_partial['TEMP'].values

        eda_min, eda_max, eda_mean, eda_std = statistical_features(eda)
        hr_min, hr_max, hr_mean, hr_std = statistical_features(hr)
        temp_min, temp_max, temp_mean, temp_std = statistical_features(temp)
        eda_skew, eda_kurtosis = shape_features(eda)

        hr_rms = calculate_rms(hr)

        peaks, properties = find_peaks(eda, width=5)
        num_Peaks = len(peaks)

        prominences = np.array(properties['prominences'])
        widths = np.array(properties['widths'])
        amplitude = np.sum(prominences)
        duration = np.sum(widths)

        df_features.loc[index] = [eda_mean, eda_min, eda_max, eda_std, eda_kurtosis, eda_skew, num_Peaks, amplitude,
                                  duration, hr_mean, hr_min, hr_max, hr_std, hr_rms, temp_mean, temp_min, temp_max, temp_std]

        index = index + 1

    return df_features

def generate_lag_features(input_df, columns, lags):
    cols = list(map(str, range(len(columns) * len(lags), 0, -1)))
    lag_df = pd.DataFrame(columns=cols)

    index = len(columns) * len(lags)

    for col in tqdm(columns, desc="Generating lag features", leave=True):
        for lag in tqdm(lags, desc=f"Lag features for {col}", leave=True):
            lagged_column = f'{index}'
            # Convert input_df to DataFrame and access the column
            lag_df[lagged_column] = pd.DataFrame(input_df)[col].shift(lag)
            index -= 1
            
    return lag_df

def scale_features(df_total, feature_columns):
    scaled_df = df_total.copy()
    scaler = MinMaxScaler()
    scaled_df[feature_columns] = scaler.fit_transform(scaled_df[feature_columns])
    scaled_df = scaled_df.fillna(0)
    return scaled_df

def get_X_r(df):
    df_features = extract_features(df)

    print(df_features)

    cols = ['HR_Mean', 'TEMP_Mean', 'EDA_Mean']
    lags = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    df_lag_features = generate_lag_features(df_features, cols, lags)

    print("df_lag_features")
    print(df_lag_features)

    df_total = pd.concat([df_lag_features, df_features], axis=1)
    print(df_total.shape)

    feature_cols = df_total.columns[:48]
    df_total_scaled_r = scale_features(df_total, feature_cols)

    print("df_total_scaled_r")
    print(df_total_scaled_r)

    X_r = df_total_scaled_r[feature_cols]
    X_r[feature_cols] = X_r[feature_cols].fillna(0)

    print(X_r)

    return X_r