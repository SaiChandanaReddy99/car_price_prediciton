import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("Dataset/Car details v3.csv")

# Data Preprocessing
def preprocess_data(df):
    """Cleans and preprocesses the dataset."""
    df = df.dropna()
    
    # Convert mileage, engine, max_power to numerical values
    df = df.copy()
    df['mileage'] = df['mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '').astype(float)
    df['engine'] = df['engine'].str.replace(' CC', '').astype(int)
    df['max_power'] = df['max_power'].str.replace(' bhp', '').astype(float)
    
    # Remove outliers using Interquartile Range (IQR)
    numerical_cols = ["selling_price", "km_driven", "mileage", "engine", "max_power"]
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    # Log transform skewed features
    df['km_driven'] = np.log1p(df['km_driven'])
    df['selling_price'] = np.log1p(df['selling_price'])
    
    # Drop unnecessary columns
    df = df.drop(columns=["name", "torque", "seats"])
    
    return df

# Preprocess the data
df = preprocess_data(df)

# Splitting Data
X = df.drop(columns=["selling_price"])
y = df["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing pipeline
numeric_features = ["year", "km_driven", "mileage", "engine", "max_power"]
categorical_features = ["fuel", "seller_type", "transmission", "owner"]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Define the Model (Gradient Boosting Regressor)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42))
])

# Train the Model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Reverse log transformation
y_pred = np.expm1(y_pred)
y_test = np.expm1(y_test)

# Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print Model Performance
print("Model Performance:")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R² Score: {r2}")

# Save the Model
joblib.dump(model, "car_price_model.pkl")

# Save Processed Data
df.to_csv("Dataset/processed_car_data.csv", index=False)
