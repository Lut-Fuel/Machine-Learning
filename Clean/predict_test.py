import numpy as np
import joblib
from keras.models import load_model

model = load_model('model.h5')
scaler = joblib.load('scaler.joblib')

sample_input = np.array([[4,    #Num of cylinder
                          0,    #Engine type
                          224.0,#Engine HP
                          5000, #Engine HP rpm
                          1,    #Transmission
                          50,   #Fuel tank capacity
                          6.2,  #Acceleration
                          3     #Fuel Grade
]])

scaled_input = scaler.transform(sample_input)

prediction = model.predict(scaled_input)
predicted_value = prediction[0][0]
print(f'Predicted value: {predicted_value}')