import pickle
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import datetime as dt

from feature_extraction import get_X_r


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StressItem(BaseModel):
    Temperature: float
    HR: float
    GSR: float

with open('stress-model-regressor.pkl', 'rb') as f:
    model = pickle.load(f)


@app.get('/')
async def get_page():
    return "okay"


# endpoint for sending single datapoint 
@app.post('/single-data')
async def test_endpoint(item: StressItem):
    # Extract features from the incoming data
    columns = ['EDA', 'HR', 'TEMP']
    df = pd.DataFrame([item.dict().values()], columns = columns)

    # Duplicate the row 10 times
    df = pd.concat([df] * 10, ignore_index=True)    
    print(df)

    X_r = get_X_r(df)

    prediction = model.predict(X_r) # pass the df here
    print(prediction)

    return {"prediction": prediction[0]}

@app.post('/csv')
def predict_from_csv(csv: UploadFile = File(...)):
    df_csv = pd.read_csv(csv.file)

    # Rename columns
    df_csv.rename(columns={
        'Object Temperature(C)': 'TEMP',
        'Heart Rate Ear(BPM)': 'HR',
        'GSR': 'EDA'
    }, inplace=True)

    # Create a new DataFrame with selected columns
    selected_columns = ['TEMP', 'HR', 'EDA']
    df = df_csv[selected_columns]

    csv.file.close()
    print(df)

    X_r = get_X_r(df)

    prediction = model.predict(X_r) # pass the df here

    # Concatenate X_r and prediction into a DataFrame
    result_df = pd.concat([X_r, pd.DataFrame({'Prediction': prediction})], axis=1)

    # Save the result DataFrame to a CSV file
    date_format = '%Y_%m_%d'  # Format for extracting only the date
    current_date_time_dt = dt.datetime.now()  # Current Date and Time in a DateTime Object.
    current_date_string = dt.datetime.strftime(current_date_time_dt, date_format) 

    result_csv_filename = f'result_{current_date_string}.csv'
    result_df.to_csv(result_csv_filename, index=False)

    return {"prediction": prediction[0]}


