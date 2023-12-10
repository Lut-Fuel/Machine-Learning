import numpy as np
import joblib
from keras.models import load_model

# Load the saved model and scaler
model = load_model('model_dinova1.h5')
scaler = joblib.load('scaler_dinova1.joblib')

# Create a sample input for testing
sample_input = np.array([[
    8,    # Number_of_Cylinders
    0,    # Engine_Type
    354.0,    # Engine_Horse_Power
    6500,    # Engine_Horse_Power_RPM
    1,    # Transmission
    5.6,  # Acceleration_0_to_100_Km
    3     # Fuel_Grade
]])

# Standardize the input data using the loaded scaler
sample_input_scaled = scaler.transform(sample_input)

# Make a prediction using the loaded model
prediction = model.predict(sample_input_scaled)

# Extract the predicted value
predicted_value = prediction[0][0]

print(f'Predicted value: {predicted_value}')