import ML
import WeatherApi


class MainApplication:
    def __init__(self):
        self.weather = WeatherApi.WeatherApi()
        self.model = ML.ML()
        self.Soil_Moisture = 0
        self.Soil_Humidity = 0

    def set_Soil_Moisture(self,Soil_moisture):
        self.Soil_Moisture = Soil_moisture

    def set_Soil_Humidity(self,Soil_Humidity):
        self.Soil_Humidity = Soil_Humidity

    def get_weather_data(self):
        self.weather.API()


    def make_prediction(self):
        input_data = self.model.set_input_data(self.Soil_Moisture, self.weather.Temperature, self.Soil_Humidity, self.weather.Time, self.weather.Air_Temperature, self.weather.Wind_Speed, self.weather.Air_Humidity)
        prediction = self.model.set_prediction()[0]
        return prediction

if __name__ == "__main__":
    app = MainApplication()
    app.get_weather_data()
    app.set_Soil_Moisture(4)
    app.set_Soil_Humidity(49)
    prediction = app.make_prediction()


    print(f"Prediction: {prediction}")









