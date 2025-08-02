import requests
import pandas as pd
from datetime import datetime, timedelta

def get_weather_and_seismic_data(lat, lon):
    try:
        # Weather API call
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&hourly=temperature_2m,relativehumidity_2m,cloudcover"
            "&daily=precipitation_sum&timezone=auto"
        )
        weather_response = requests.get(weather_url)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        now = datetime.now()
        hour_idx = now.hour

        temperature = weather_data.get("hourly", {}).get("temperature_2m", [None]*24)[hour_idx]
        rainfall = weather_data.get("daily", {}).get("precipitation_sum", [None])[0]
        cloud_cover = weather_data.get("hourly", {}).get("cloudcover", [None]*24)[hour_idx]
        humidity = weather_data.get("hourly", {}).get("relativehumidity_2m", [None]*24)[hour_idx]

        # Seismic API call
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=7)
        seismic_url = (
            f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
            f"&starttime={start_time.strftime('%Y-%m-%dT%H:%M:%S')}"
            f"&endtime={end_time.strftime('%Y-%m-%dT%H:%M:%S')}"
            f"&latitude={lat}&longitude={lon}&maxradiuskm=200"
        )
        seismic_response = requests.get(seismic_url)
        seismic_response.raise_for_status()
        seismic_data = seismic_response.json()
        seismic_count = len(seismic_data.get("features", []))

        return {
            "Date": now.strftime("%Y-%m-%d"),
            "Temperature_Celsius": temperature,
            "Rainfall_mm": rainfall,
            "Cloud_Cover_Percent": cloud_cover,
            "Humidity_Percent": humidity,
            "Seismic_Event_Count": seismic_count
        }

    except Exception as e:
        return {
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Temperature_Celsius": None,
            "Rainfall_mm": None,
            "Cloud_Cover_Percent": None,
            "Humidity_Percent": None,
            "Seismic_Event_Count": 0,
            "Error": str(e)
        }

if __name__ == "__main__":
    print("📁 Please enter the path to your CSV file with columns: GLIMS_ID, Longitude, Latitude")
    file_path = input("CSV file path: ").strip()

    try:
        df = pd.read_csv(file_path)
        results = []

        for idx, row in df.iterrows():
            glims_id = row["GLIMS_ID"]
            lon = row["Longitude"]
            lat = row["Latitude"]
            print(f"🔄 Fetching data for {glims_id} at ({lat}, {lon})...")

            data = get_weather_and_seismic_data(lat, lon)
            results.append({
                "GLIMS_ID": glims_id,
                "Longitude": lon,
                "Latitude": lat,
                **data
            })

        output_df = pd.DataFrame(results)
        output_file = "glims_output_with_weather_seismic.csv"
        output_df.to_csv(output_file, index=False)
        print(f"\n✅ Output saved to: {output_file}")

    except FileNotFoundError:
        print("❌ File not found. Please check the file path and try again.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
