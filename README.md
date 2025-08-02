# glofi
This app calculates GLOF intensity from data ingested from a CSV file. The code was orginally created by Hritesh Ghosh.

The Glacier.csv files contains information of Glaciers in and impacting Bhutan.

The GET_DATA.py ingests Glacier.csv and connects to weather APIs to gather weather data for the glaciers in the csv file. It then produces a new csv file with this current weather data appended to the CSV (adds the appropriate fields for each record).

The nglof.py ingests the csv file produced in the previous stage and calculates GLOF intensity for each Glacier and maps this through a streamlit app.
