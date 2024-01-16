import pickle
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
from api.feature_extraction import get_X_r


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","https://burnout-sentinels.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StressItem(BaseModel):
    Temperature: float
    HR: float
    GSR: float

with open('api/stress-model-regressor.pkl', 'rb') as f:
    model1 = pickle.load(f)

with open('api/stress-classifier-model.pkl', 'rb') as f:
    model2 = pickle.load(f)


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
    df = pd.concat([df] * 40, ignore_index=True)    
    print(df)

    X_r = get_X_r(df)

    prediction = model1.predict(X_r) # pass the df here
    print(prediction)

    return {"prediction": prediction[0]}


@app.post('/csv')
async def predict_from_csv(csv: UploadFile = File(...)):
    try:
        print("hereeeeeees")

        # Check file extension
        file_extension = csv.filename.split('.')[-1].lower()

        # Read data based on the file extension
        if file_extension == 'csv':
            df_csv = pd.read_csv(csv.file)
        elif file_extension in ['xlsx', 'xls']:
            print("detected xlsx or xls")
            # df_csv = pd.read_excel(csv.file)
            excel_content = io.BytesIO(csv.file.read())
            df_csv = pd.read_excel(excel_content)
        else:
            return {"ok": False, "message": "Inappropriate file extension"}
        print("File read.")

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

        X_r = get_X_r(df)

        prediction = model1.predict(X_r)  # pass the df here
        average_prediction = np.mean(prediction)

        classification = model2.predict(X_r)
        print(prediction)
        rounded_classification = round(np.mean(classification))

        # Mapping dictionary for stress levels
        stress_mapping = {
            0: 'Low',
            1: 'Medium',
            2: 'High'
        }

        # Lookup the stress level based on the rounded classification
        stress_level = stress_mapping.get(rounded_classification, 'Unknown Stress Level')

        print(average_prediction)
        print(stress_level)

        return {"prediction": float(average_prediction), "classification": stress_level}

    except Exception as e:
        raise HTTPException(status_code=500, detail="Oh no, something went wrong!")



