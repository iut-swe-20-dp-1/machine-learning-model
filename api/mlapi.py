from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

class StressItem(BaseModel):
    Temperature: float
    HR: int
    GSR: int

with open('Regressor_Date_Time_2024_01_13.pkl', 'rb') as f:
    model = pickle.load(f)

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

def generate_lag_features(input_df, columns, lags):
    cols = list(map(str, range(len(columns) * len(lags), 0, -1)))
    lag_df = pd.DataFrame(columns=cols)

    index = len(columns) * len(lags)

    for col in tqdm(columns, desc="Generating lag features", leave=True):
        for lag in tqdm(lags, desc=f"Lag features for {col}", leave=True):
            lagged_column = f'{index}'
            lag_df[lagged_column] = input_df[col].shift(lag)
            index -= 1
            
    return lag_df


@app.post('/')
async def test_endpoint(item: StressItem):
    # Extract features from the incoming data
    columns = ['EDA', 'HR', 'TEMP']
    df = pd.DataFrame([item.dict().values()], columns = columns)
    print(df)

    cols = [
        'EDA_Mean', 'EDA_Min', 'EDA_Max', 'EDA_Std', 'EDA_Kurtosis', 'EDA_Skew', 'EDA_Num_Peaks', 'EDA_Amplitude', 'EDA_Duration',
        'HR_Mean', 'HR_Min', 'HR_Max', 'HR_Std', 'HR_RMS', 'TEMP_Mean', 'TEMP_Min', 'TEMP_Max', 'TEMP_Std'
    ]

    eda = df['EDA'].values
    hr = df['HR'].values
    temp = df['TEMP'].values

    df_features = pd.DataFrame(columns=cols)
    index = 0

    eda_min, eda_max, eda_mean, eda_std = statistical_features(eda)
    hr_min, hr_max, hr_mean, hr_std = statistical_features(hr)
    temp_min, temp_max, temp_mean, temp_std = statistical_features(temp)
    eda_skew, eda_kurtosis = shape_features(eda)
    
    hr_rms = calculate_rms(hr)
    temp_rms = calculate_rms(temp)

    peaks, properties = find_peaks(eda, width=5)
    num_Peaks = len(peaks)

    prominences = np.array(properties['prominences'])
    widths = np.array(properties['widths'])
    amplitude = np.sum(prominences)
    duration = np.sum(widths)

    

    prediction = model.predict() # pass the df here

    return {"prediction": prediction}


