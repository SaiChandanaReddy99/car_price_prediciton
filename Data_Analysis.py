import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Dataset/Car details v3.csv")

# Data Preprocessing
def preprocess_data(df):
    df = df.dropna()
    
    # Convert mileage, engine, max_power to numerical
    df = df.copy()
    df['mileage'] = df['mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '').astype(float)
    df['engine'] = df['engine'].str.replace(' CC', '').astype(int)
    df['max_power'] = df['max_power'].str.replace(' bhp', '').astype(float)
    
    # Drop unnecessary columns
    df = df.drop(columns=["name", "torque"])
    
    return df

df = preprocess_data(df)

# Convert categorical variables to numeric for correlation heatmap
df_encoded = df.copy()
categorical_columns = ["fuel", "seller_type", "transmission", "owner"]
df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns, drop_first=True)

# Data Analysis
print("Basic Statistics:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

# Pairplot
sns.pairplot(df_encoded, diag_kind="kde")
plt.show()

# Distribution of Selling Price
plt.figure(figsize=(8, 5))
sns.histplot(df["selling_price"], bins=30, kde=True)
plt.title("Distribution of Selling Price")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Save processed data
df.to_csv("Dataset/processed_car_data.csv", index=False)
