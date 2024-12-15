from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd


class ML:
    def __init__(self, path='Watering_model.joblib'):
        self.prediction = 0
        self.input_data = []
        self.ml = joblib.load(path)
        self.feature_names = [
            'Soil Moisture', 'Temperature', ' Soil Humidity', 'Time',
            'Air temperature (C)', 'Wind speed (Km/h)', 'Air humidity (%)'
        ]

    def set_input_data(self, Soil_Moisture, Temperature, Soil_Humidity, Time, Air_temperature, Wind_speed,
                       Air_humidity):
        self.input_data = [Soil_Moisture, Temperature, Soil_Humidity, Time, Air_temperature, Wind_speed, Air_humidity]
        return self.input_data

    def set_prediction(self):
        df_input_data = pd.DataFrame([self.input_data], columns=self.feature_names)
        self.prediction = self.ml.predict(df_input_data)
        return self.prediction


# Example Usage
if __name__ == "__main__":
    model = ML()
    input_data = model.set_input_data(30, 0, 40, 12, 28, 5, 60)
    prediction = model.set_prediction()[0]
    print(f"Prediction: {prediction}")
