import requests
import json
import pandas as pd

API_key = 'c6fb64fec365de60c2e15986a29b3a1c'
lat = 18.520430
lon = 73.856743
url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_key}"
response = requests.get(url)

data=response.json()

print(json.dumps(data,indent=3))

weather_data = {
    'City': data.get('name'),
    'Country': data.get('sys', {}).get('country'),
    'Latitude': data.get('coord', {}).get('lat'),
    'Longitude': data.get('coord', {}).get('lon'),
    'Temperature (K)': data.get('main', {}).get('temp'),
    'Feels Like (K)': data.get('main', {}).get('feels_like'),
    'Weather Description': data.get('weather', [{}])[0].get('description'),
    'Humidity (%)': data.get('main', {}).get('humidity'),
    'Pressure (hPa)': data.get('main', {}).get('pressure'),
    'Wind Speed (m/s)': data.get('wind', {}).get('speed'),
    'Cloudiness (%)': data.get('clouds', {}).get('all')
}

df = pd.DataFrame([weather_data])

print(df)