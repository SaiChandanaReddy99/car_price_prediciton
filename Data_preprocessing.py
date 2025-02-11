import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Dataset/Car details v3.csv")

# Data Preprocessing
def preprocess_data(df):
    df = df.dropna()
    
    # Convert mileage, engine, max_power to numerical
    df['mileage'] = df['mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '').astype(float)
    df['engine'] = df['engine'].str.replace(' CC', '').astype(int)
    df['max_power'] = df['max_power'].str.replace(' bhp', '').astype(float)
    
    # Drop unnecessary columns
    df = df.drop(columns=["name", "torque"])
    
    return df

df = preprocess_data(df)

# Save processed data
df.to_csv("Dataset/processed_car_data.csv", index=False)
