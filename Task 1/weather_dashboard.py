import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime 

#configuration
API_KEY = "d90d03a58c055087368a6a3ad74bf100"
CITY = "Kochi"
UNITS = "metric"
URL = "http://api.openweathermap.org/data/2.5/forecast"

def fetch_data():
    params = {
        "q":CITY,
        "appid":API_KEY,
        "units":UNITS
    }

    try:
        response = requests.get(URL,params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        exit()

def process_data(json_data):
    forecasts = json_data["list"]
    processed_data = []
    for forecast in forecasts:
        processed_data.append({
            "datetime": datetime.utcfromtimestamp(forecast["dt"]),
            "temperature": forecast["main"]["temp"],
            "humidity": forecast["main"]["humidity"],
            "wind_speed": forecast["wind"]["speed"],
            "weather": forecast["weather"][0]["main"]
        })

    return pd.DataFrame(processed_data)

def create_dashboard(df):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12,10))

    #temperature plot
    plt.subplot(3,1,1)
    sns.lineplot(data=df,x="datetime",y="temperature",color="tomato")
    plt.title(f"Weather forecast for {CITY}",fontweight="bold",pad=20)
    plt.xlabel("")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)

    #humidity plot
    plt.subplot(3,1,2)
    sns.barplot(data=df,x="datetime",y="humidity",color="steelblue")
    plt.xlabel("")
    plt.ylabel("Humidity(%)")
    plt.xticks(rotation=45)

    #wind speed plot
    plt.subplot(3,1,3)
    sns.lineplot(data=df,x="datetime",y="wind_speed",color="seagreen")
    plt.xlabel("Date and time")
    plt.ylabel("Wind speed(m/s)")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("weather_dashboard.png")
    plt.show()

if __name__ == "__main__":
    #fetch and process data
    raw_data = fetch_data()
    weather_df = process_data(raw_data)

    #generate dashboard
    create_dashboard(weather_df)