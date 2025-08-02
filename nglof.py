import streamlit as st
import pandas as pd
import numpy as np
import re

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
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", coord_str)
    if len(numbers) >= 2:
        return f"{numbers[0]}-{numbers[1]}"
    return coord_str

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
    df['effective_temp'] = np.maximum(df['temperature'], 0)
    
    # Dam risk weights
    dam_risk = {'moraine': 0.8, 'ice': 0.5, 'bedrock': 0.2}
    df['dam_risk'] = df['dam_type'].map(dam_risk)
    
    # Convert lake_volume from liters to cubic meters (1 m³ = 1000 L)
    df['lake_volume_m3'] = df['lake_volume'] / 1000
    
    # Calculate lake area from volume using inverse scaling
    df['lake_area_km2'] = df.apply(
        lambda row: calculate_lake_area(row['lake_volume_m3'], row['dam_type']), 
        axis=1
    )
    
    # Convert ice_coverage from percentage to decimal (100% → 1.0)
    df['ice_coverage'] = df['ice_coverage'].str.rstrip('%').astype(float) / 100.0
    
    # Physics model components
    df['melt_water'] = melt_coeff * df['effective_temp'] * df['ice_coverage']
    df['total_water'] = df['lake_volume_m3'] + df['melt_water'] + 0.8 * df['rainfall']
    df['slope_factor'] = np.sqrt(np.tan(np.radians(df['slope_gradient'])))
    
    # Base intensity calculation
    intensity = base_factor * df['total_water'] * df['slope_factor'] * df['dam_risk']
    
    # Environmental modifiers
    intensity *= (1 + 0.1 * (df['elevation'] / 1000))
    intensity *= (1 + 0.05 * df['seismic_activity'])
    intensity *= (1 + 0.01 * (df['humidity'] - 70))
    intensity *= (1 - 0.002 * df['cloud_cover'])
    
    return np.round(intensity, 2)

def classify_intensity(intensity_series):
    """Classify intensity into risk categories with updated thresholds"""
    return pd.cut(intensity_series, 
                  bins=INTENSITY_THRESHOLDS + [float('inf')], 
                  labels=INTENSITY_LABELS, 
                  right=False)

# Streamlit app
st.title("🌊 Glacial Lake Outburst Flood Risk Assessment")
st.write("Upload a CSV file with glacial lake data to calculate flood risk intensity")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read uploaded file
        df = pd.read_csv(uploaded_file)
        
        # Check required columns
        required_columns = {
            'temperature', 'ice_coverage', 'lake_volume', 'rainfall',
            'slope_gradient', 'dam_type', 'elevation', 'seismic_activity',
            'humidity', 'cloud_cover', 'location_id', 'date'
        }
        
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()
        
        # Clean coordinates - handle formats like "(27.96 - 90.594)"
        st.info("Cleaning location coordinates...")
        df['original_location'] = df['location_id']  # Preserve original
        df['location_id'] = df['location_id'].apply(clean_coordinate)
        
        # Process data
        with st.spinner('Calculating flood intensities...'):
            df['intensity'] = calculate_intensity(df)
            df['intensity_class'] = classify_intensity(df['intensity'])
        
        # Display results
        st.success("Processing complete!")
        
        # Show statistics
        st.subheader("Risk Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Max Intensity", f"{df['intensity'].max():,.2f}")
        col2.metric("Min Intensity", f"{df['intensity'].min():,.2f}")
        col3.metric("Avg Intensity", f"{df['intensity'].mean():,.2f}")
        
        # Classification distribution
        st.subheader("Risk Classification Distribution")
        class_counts = df['intensity_class'].value_counts()
        st.bar_chart(class_counts)
        
        # Show high-risk samples
        st.subheader("High-Risk Locations")
        high_risk = df[df['intensity_class'] == 'Catastrophic']
        if not high_risk.empty:
            # Format location coordinates for display
            high_risk_display = high_risk.copy()
            high_risk_display['Coordinates'] = high_risk_display['original_location'].apply(
                lambda x: f"({', '.join(re.findall(r'[-+]?\d*\.\d+|\d+', x)[:2])})"
            )
            st.dataframe(high_risk_display[['Coordinates', 'date', 'dam_type', 'lake_volume', 'intensity', 'intensity_class']].head(10))
        else:
            st.info("No catastrophic risk locations found")
        
        # Show processed data
        st.subheader("Processed Data Preview")
        df_display = df.copy()
        df_display['Coordinates'] = df_display['original_location'].apply(
            lambda x: f"({', '.join(re.findall(r'[-+]?\d*\.\d+|\d+', x)[:2])})"
        )
        st.dataframe(df_display[['Coordinates', 'date', 'dam_type', 'lake_volume', 'intensity', 'intensity_class']].head())
        
        # Download button
        st.download_button(
            label="Download processed data",
            data=df.drop(columns=['original_location']).to_csv(index=False).encode('utf-8'),
            file_name='glof_risk_processed.csv',
            mime='text/csv'
        )
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.error("Please ensure your data format matches the requirements")
else:
    st.info("Please upload a CSV file to get started")
    st.markdown("""
    **Sample data should include these columns:**
    - `temperature` (in °C)
    - `ice_coverage` (percentage - values like 100%)
    - `lake_volume` (in liters)
    - `rainfall` (mm)
    - `slope_gradient` (degrees)
    - `dam_type` (moraine/ice/bedrock)
    - `elevation` (meters)
    - `seismic_activity` (Richter scale)
    - `humidity` (percentage)
    - `cloud_cover` (percentage)
    - `location_id` (coordinates in formats like "(27.96 - 90.594)")
    - `date` (YYYY-MM-DD)
    
    **Volume-Area Scaling Calculation:**
    The app uses inverse volume-area scaling to calculate lake area from volume:
    - For ice dams: A = (V / 0.0538)<sup>1/1.25</sup>
    - For moraine/bedrock dams: A = (V / 0.0365)<sup>1/1.375</sup>
    
    Where:
    - V = Lake volume (km³)
    - A = Lake surface area (km²)
    
    **Risk Classification:**
    - Intensity = 0 → **None**
    - 500 ≤ Intensity < 2000 → **Minor**
    - 2000 ≤ Intensity < 5000 → **Moderate**
    - 5000 ≤ Intensity < 5500 → **Major**
    - Intensity ≥ 5500 → **Catastrophic**
    
    **The app automatically:**
    - Converts lake volume from liters to m³ (÷1000) then to km³ (÷1e9)
    - Converts ice coverage from percentage to decimal (100% → 1.0)
    - Calculates lake area using the inverse scaling equations based on dam type
    - Handles coordinate cleaning (parentheses and spaces)
    - Classifies risk based on calculated intensity
    """, unsafe_allow_html=True)
