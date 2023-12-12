import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from pydantic import BaseModel
import pickle

# Create an app object
app = FastAPI()

# Loads model, scaler, and transformer
model = load_model('car_regress.h5')
scaler = joblib.load('scaler.joblib')
transformer = joblib.load('transformer.joblib')

# Class that describes the car data formats
class carData(BaseModel):
    Number_of_Cylinders : int
    Engine_Type : int
    Engine_Horse_Power : float
    Engine_Horse_Power_RPM : float
    Transmission : int
    Acceleration_0_to_100_Km : float
    Fuel_Grade : int
    

# Main page of the web app
@app.get('/')
def index():
    return {'Welcome to the browser app'}

# Prediction page
@app.post('/car/predict')
def predict_fuel(datas:carData):
    data = datas.dict()
    print(data)
    Number_of_Cylinders = data['Number_of_Cylinders']
    Engine_Type = data['Engine_Type']
    Engine_Horse_Power = data['Engine_Horse_Power']
    Engine_Horse_Power_RPM = data['Engine_Horse_Power_RPM']
    Transmission = data['Transmission']
    Acceleration_0_to_100_Km = data['Acceleration_0_to_100_Km']
    Fuel_Grade = data['Fuel_Grade']
    scaled_columns = ['Engine_Horse_Power',
                      'Engine_Horse_Power_RPM',
                      'Acceleration_0_to_100_Km'
                      ]
    data_test = [[
        Number_of_Cylinders,
        Engine_Type,
        Transmission,
        Fuel_Grade,
        Engine_Horse_Power,
        Engine_Horse_Power_RPM,
        Acceleration_0_to_100_Km
    ]]
    data_test = pd.DataFrame(data_test, columns=['Number_of_Cylinders', 
                                                            'Engine_Type',
                                                            'Transmission',
                                                            'Fuel_Grade',
                                                            'Engine_Horse_Power',
                                                            'Engine_Horse_Power_RPM',
                                                            'Acceleration_0_to_100_Km'])
    data_test[scaled_columns] = scaler.transform(data_test[scaled_columns])
    data_test = transformer.transform(data_test)
    data_test= tf.data.Dataset.from_tensor_slices(data_test)
    data_test = data_test.batch(1)
    hasil = model.predict(data_test)[0][0]
    return {
            'Hasil prediksi bahan bakar (dalam Km/L)': round(hasil.tolist(), 2)
    }

# Run the API using uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    

