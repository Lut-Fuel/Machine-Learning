import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

#Import training datas
main_data = pd.read_csv('Dataset/train_data_cleaned.csv')
indo_data = pd.read_csv('Dataset/indo_car_train_data.csv')

#Combine the main training data with Indonesian car data
df = pd.concat([main_data, indo_data], axis=0, ignore_index=True)
df.dropna(inplace=True)

#Define the features and target
target = df['Mixed_Fuel_Consumption_per_100_km_l']

selected_columns = ['Number_of_Cylinders',
                    'Engine_Type',
                    'Engine_Horse_Power',
                    'Engine_Horse_Power_RPM',
                    'Transmission',
                    'Fuel_Tank_Capacity',
                    'Acceleration_0_to_100_Km',
                    'Fuel_Grade']

features = df[selected_columns]

#Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=42)

#Initialize the scaler for the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Initialize the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mae'])

#Create the history function to save logs
class SaveHistoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open('training_history.txt', 'a') as f:
            f.write(f'Epoch {epoch + 1}/{self.params["epochs"]} - '
                    f'Loss: {logs["loss"]:.4f} - '
                    f'MAE: {logs["mae"]:.4f} - '
                    f'Val. Loss: {logs["val_loss"]:.4f} - '
                    f'Val. MAE: {logs["val_mae"]:.4f}\n')
        
    def on_train_end(self, logs=None):
        date_time = datetime.now()
        with open('training_history.txt', 'a') as f:
            f.write(f'Training done at {date_time}\n'
                    f'\n')

save_history_callback = SaveHistoryCallback()

#Train the model
model.fit(X_train_scaled,
          y_train,
          epochs=5000,
          batch_size=1024,
          validation_data=(X_test_scaled, y_test),
          callbacks=[save_history_callback])

#Saving model for deployment
model.save('model.h5')
joblib.dump(scaler, 'scaler.joblib')