from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import joblib
from keras.models import load_model

# Load the saved model and scaler
model = load_model('model_dinova1.h5')
scaler = joblib.load('scaler_dinova1.joblib')

# Define the FastAPI app
app = FastAPI()

# Define Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Define a Pydantic model for the request payload
class InputData(BaseModel):
    Number_of_Cylinders: int
    Engine_Type: int
    Engine_Horse_Power: float
    Engine_Horse_Power_RPM: int
    Transmission: int
    Acceleration_0_to_100_Km: float
    Fuel_Grade: int

# Define a Pydantic model for the response
class PredictionResponse(BaseModel):
    predicted_value: float
    result: float

# Define a route for rendering the form
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define a route for handling form submission
@app.post("/predict", response_model=PredictionResponse)  # Specify the response model
async def predict(data: InputData):
    # Convert the input data to a NumPy array
    input_data = np.array([[
        data.Number_of_Cylinders,
        data.Engine_Type,
        data.Engine_Horse_Power,
        data.Engine_Horse_Power_RPM,
        data.Transmission,
        data.Acceleration_0_to_100_Km,
        data.Fuel_Grade
    ]])

    # Standardize the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)

    # Make a prediction using the loaded model
    prediction = model.predict(input_data_scaled)

    # Extract the predicted value
    predicted_value = prediction[0][0]

    # Define the price of fuel (replace with your actual price)
    fuel_price = 10000

    # Multiply the predicted value by the price of fuel
    result = predicted_value * fuel_price

    return {"predicted_value": predicted_value, "result": result}