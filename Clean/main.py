import tensorflow as tf
import numpy as np
import uvicorn
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

model = tf.keras.models.load_model('model.h5')
scaler = joblib.load('scaler.joblib')

app = FastAPI()

class UserInput(BaseModel):
    Number_of_Cylinders: int
    Engine_Type: int
    Engine_Horse_Power: float
    Engine_Horse_Power_RPM: int
    Transmission: int
    Fuel_Tank_Capacity: int
    Acceleration_0_to_100_Km: float
    Fuel_Grade: int

class PredictionResponse(BaseModel):
    prediction: float

@app.get('/')
async def index():
    return {"Message": "Welcome to Lut-Fuel"}

@app.post('/predict/', response_model=PredictionResponse)
async def predict(data: UserInput):
    input_data = np.array([[
        data.Number_of_Cylinders,
        data.Engine_Type,
        data.Engine_Horse_Power,
        data.Engine_Horse_Power_RPM,
        data.Transmission,
        data.Fuel_Tank_Capacity,
        data.Acceleration_0_to_100_Km,
        data.Fuel_Grade
    ]])
    
    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)

    predicted_value = prediction[0][0]

    return {"prediction": predicted_value}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)