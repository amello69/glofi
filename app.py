import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime, timedelta, UTC

# --- Functions from GET_DATA.py ---
def get_weather_and_seismic_data(lat, lon):
    """
    Fetches current weather and recent seismic data for a given latitude and longitude.
    
    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        
    Returns:
        dict: A dictionary with weather and seismic data, or an error message.
    """
    try:
        # Weather API call (Open-Meteo)
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

        # Seismic API call (USGS) for the last 7 days within 200km
        end_time = datetime.now(UTC)
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

# --- Functions from nglof.py ---
# Updated intensity thresholds and labels
INTENSITY_THRESHOLDS = [0, 500, 2000, 5000, 5500]
INTENSITY_LABELS = ['None', 'Minor', 'Moderate', 'Major', 'Catastrophic']

# Dam type parameters for volume-area scaling
DAM_TYPE_PARAMS = {
    'moraine': {'c': 0.0365, 'gamma': 1.375},
    'ice': {'c': 0.0538, 'gamma': 1.25},
    'bedrock': {'c': 0.0365, 'gamma': 1.375}
}

def clean_coordinate(coord_str):
    """Extract numeric values from coordinate string with parentheses and spaces"""
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(coord_str))
    if len(numbers) >= 2:
        return f"{numbers[0]}-{numbers[1]}"
    return str(coord_str)

def calculate_lake_area(volume_m3, dam_type):
    """
    Calculate lake area from volume using inverse volume-area scaling equation
    A = (V / c)^(1/γ)
    Where:
      V: lake volume (m³)
      A: lake surface area (km²)
      c: scaling coefficient
      γ: scaling exponent
    """
    volume_km3 = volume_m3 / 1e9  # Convert m³ to km³
    params = DAM_TYPE_PARAMS.get(dam_type.lower(), DAM_TYPE_PARAMS['moraine'])
    c = params['c']
    gamma = params['gamma']
    area_km2 = (volume_km3 / c) ** (1/gamma)
    return area_km2

def calculate_intensity(df):
    """Calculate flood intensity using physics-based model with volume-area scaling"""
    # Physics parameters
    melt_coeff = 0.15
    base_factor = 2.7
    
    # Feature engineering
    df['effective_temp'] = np.maximum(df['Temperature_Celsius'], 0)
    
    # Dam risk weights
    dam_risk = {'moraine': 0.8, 'ice': 0.5, 'bedrock': 0.2}
    # Using 'dam_type' from the user's data, mapping to risk
    df['dam_risk'] = df['dam_type'].str.lower().map(dam_risk)
    
    # Ensure dam_risk is not NaN (default to moraine risk if type is unknown)
    df['dam_risk'] = df['dam_risk'].fillna(dam_risk['moraine'])
    
    # Convert lake_volume from liters to cubic meters (1 m³ = 1000 L)
    df['lake_volume_m3'] = df['lake_volume'] / 1000
    
    # Calculate lake area from volume using inverse scaling
    df['lake_area_km2'] = df.apply(
        lambda row: calculate_lake_area(row['lake_volume_m3'], row['dam_type']), 
        axis=1
    )
    
    # Convert ice_coverage from percentage to decimal (100% → 1.0)
    # This column might not be in the initial Glacier.csv, but should be in the enriched one
    if 'ice_coverage' in df.columns:
        df['ice_coverage'] = df['ice_coverage'].str.rstrip('%').astype(float) / 100.0
    else:
        # Assuming a default value if not present
        df['ice_coverage'] = 0.5 # Default to 50% ice coverage
        st.warning("`ice_coverage` column not found. Using a default of 50%.")
    
    # Physics model components
    df['melt_water'] = melt_coeff * df['effective_temp'] * df['ice_coverage']
    df['total_water'] = df['lake_volume_m3'] + df['melt_water'] + 0.8 * df['Rainfall_mm']
    df['slope_factor'] = np.sqrt(np.tan(np.radians(df['Slope_Mean(degree)'])))
    
    # Base intensity calculation
    intensity = base_factor * df['total_water'] * df['slope_factor'] * df['dam_risk']
    
    # Environmental modifiers
    intensity *= (1 + 0.1 * (df['Elev_Mean(m)'] / 1000))
    intensity *= (1 + 0.05 * df['Seismic_Event_Count'])
    intensity *= (1 + 0.01 * (df['Humidity_Percent'] - 70))
    intensity *= (1 - 0.002 * df['Cloud_Cover_Percent'])
    
    return np.round(intensity, 2)

def classify_intensity(intensity_series):
    """Classify intensity into risk categories with updated thresholds"""
    return pd.cut(intensity_series, 
                  bins=INTENSITY_THRESHOLDS + [float('inf')], 
                  labels=INTENSITY_LABELS, 
                  right=False)

# --- Streamlit UI ---
st.set_page_config(
    page_title="GLOF Risk Assessment App",
    layout="wide"
)

st.title("🌊 Glacial Lake Outburst Flood Risk Assessment")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["1. Fetch Data", "2. Assess GLOF Risk"])

# Initialize session state for dataframes
if 'uploaded_df_stage1' not in st.session_state:
    st.session_state['uploaded_df_stage1'] = None
if 'enriched_df' not in st.session_state:
    st.session_state['enriched_df'] = None
if 'processed_df' not in st.session_state:
    st.session_state['processed_df'] = None

# Stage 1: Data Fetching and Enrichment
if page == "1. Fetch Data":
    st.header("Stage 1: Fetch Current Weather & Seismic Data")
    st.markdown("""
        Upload your glacier CSV file. The app will fetch real-time weather and recent seismic data 
        for each glacier's location and append it to your data.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file (e.g., Glacier.csv)", type="csv")

    if uploaded_file is not None:
        try:
            st.session_state['uploaded_df_stage1'] = pd.read_csv(uploaded_file)
            st.write("Original Data Preview:")
            st.dataframe(st.session_state['uploaded_df_stage1'].head())
            
            if st.button("Process and Enrich Data"):
                required_cols = {'GLIMS_ID', 'Longitude', 'Latitude'}
                if not required_cols.issubset(st.session_state['uploaded_df_stage1'].columns):
                    st.error(f"Missing required columns: {required_cols - set(st.session_state['uploaded_df_stage1'].columns)}")
                else:
                    results = []
                    with st.spinner('Fetching data from APIs... This may take a few moments.'):
                        for _, row in st.session_state['uploaded_df_stage1'].iterrows():
                            glims_id = row["GLIMS_ID"]
                            lon = row["Longitude"]
                            lat = row["Latitude"]
                            
                            data = get_weather_and_seismic_data(lat, lon)
                            results.append({
                                "GLIMS_ID": glims_id,
                                "Longitude": lon,
                                "Latitude": lat,
                                **data
                            })
                    
                    enriched_data = pd.DataFrame(results)
                    st.session_state['enriched_df'] = pd.merge(st.session_state['uploaded_df_stage1'], enriched_data, on=['GLIMS_ID', 'Longitude', 'Latitude'], how='left')
                    st.success("Data fetching complete!")

    if st.session_state['enriched_df'] is not None:
        st.write("Enriched Data Preview:")
        st.dataframe(st.session_state['enriched_df'].head())
        csv_data = st.session_state['enriched_df'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Enriched CSV",
            data=csv_data,
            file_name='glacier_data_enriched.csv',
            mime='text/csv'
        )

# Stage 2: GLOF Risk Assessment
elif page == "2. Assess GLOF Risk":
    st.header("Stage 2: Calculate GLOF Risk & Visualize")
    
    # Check if enriched data from Stage 1 is available
    if st.session_state['enriched_df'] is not None:
        st.success("Using enriched data from Stage 1. Click 'Calculate GLOF Risk' to proceed.")
        df = st.session_state['enriched_df']
        st.write("Enriched Data Preview:")
        st.dataframe(df.head())
        
        if st.button("Calculate GLOF Risk"):
            # Check required columns
            required_columns_for_model = {
                'Temperature_Celsius', 'Rainfall_mm', 'Cloud_Cover_Percent', 'Humidity_Percent',
                'Seismic_Event_Count', 'lake_volume', 'dam_type', 'ice_coverage',
                'Slope_Mean(degree)', 'Elev_Mean(m)'
            }
            
            if not required_columns_for_model.issubset(df.columns):
                missing = required_columns_for_model - set(df.columns)
                st.error(f"Missing required columns for risk calculation: {', '.join(missing)}")
                st.info("Please ensure your enriched data contains these columns.")
            else:
                with st.spinner('Calculating flood intensities...'):
                    st.session_state['processed_df'] = df.copy()
                    st.session_state['processed_df']['intensity'] = calculate_intensity(st.session_state['processed_df'])
                    st.session_state['processed_df']['intensity_class'] = classify_intensity(st.session_state['processed_df']['intensity'])
                
                st.success("Processing complete!")
                
    else:
        st.markdown("""
            No enriched data found from Stage 1. Please upload a CSV file to calculate the GLOF risk.
        """)
        uploaded_file = st.file_uploader("Choose the enriched CSV file", type="csv")

        if uploaded_file is not None:
            try:
                st.session_state['uploaded_df_stage2'] = pd.read_csv(uploaded_file)
                st.write("Uploaded Data Preview:")
                st.dataframe(st.session_state['uploaded_df_stage2'].head())
                
                if st.button("Calculate GLOF Risk"):
                    df = st.session_state['uploaded_df_stage2']
                    required_columns_for_model = {
                        'Temperature_Celsius', 'Rainfall_mm', 'Cloud_Cover_Percent', 'Humidity_Percent',
                        'Seismic_Event_Count', 'lake_volume', 'dam_type', 'ice_coverage',
                        'Slope_Mean(degree)', 'Elev_Mean(m)'
                    }
                    
                    if not required_columns_for_model.issubset(df.columns):
                        missing = required_columns_for_model - set(df.columns)
                        st.error(f"Missing required columns for risk calculation: {', '.join(missing)}")
                        st.info("Please ensure your uploaded CSV contains these columns.")
                    else:
                        with st.spinner('Calculating flood intensities...'):
                            st.session_state['processed_df'] = df.copy()
                            st.session_state['processed_df']['intensity'] = calculate_intensity(st.session_state['processed_df'])
                            st.session_state['processed_df']['intensity_class'] = classify_intensity(st.session_state['processed_df']['intensity'])
                        
                        st.success("Processing complete!")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.error("Please ensure your data contains all required columns for the model.")

    if st.session_state['processed_df'] is not None:
        # Show statistics
        st.subheader("Risk Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Max Intensity", f"{st.session_state['processed_df']['intensity'].max():,.2f}")
        col2.metric("Min Intensity", f"{st.session_state['processed_df']['intensity'].min():,.2f}")
        col3.metric("Avg Intensity", f"{st.session_state['processed_df']['intensity'].mean():,.2f}")
        
        # Classification distribution
        st.subheader("Risk Classification Distribution")
        class_counts = st.session_state['processed_df']['intensity_class'].value_counts()
        st.bar_chart(class_counts)
        
        # Show high-risk samples
        st.subheader("High-Risk Locations")
        high_risk = st.session_state['processed_df'][st.session_state['processed_df']['intensity_class'] == 'Catastrophic']
        if not high_risk.empty:
            st.dataframe(high_risk[['GLIMS_ID', 'Date', 'dam_type', 'lake_volume', 'intensity', 'intensity_class']].head(10))
        else:
            st.info("No catastrophic risk locations found")
        
        # Show processed data
        st.subheader("Processed Data Preview")
        st.dataframe(st.session_state['processed_df'][['GLIMS_ID', 'Date', 'dam_type', 'lake_volume', 'intensity', 'intensity_class']].head())
        
        # Download button for the final processed data
        st.download_button(
            label="Download Final Processed Data",
            data=st.session_state['processed_df'].to_csv(index=False).encode('utf-8'),
            file_name='glof_risk_processed.csv',
            mime='text/csv'
        )
