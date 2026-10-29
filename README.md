
import os
import json
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Create project directories
for folder in [
    "data", "reports/figures", "models"
]:
    os.makedirs(folder, exist_ok=True)

# Step 1: Download dataset
url = (
    "https://raw.githubusercontent.com/"
    "mwaskom/seaborn-data/master/diamonds.csv"
)

file_path = "data/diamonds.csv"

if not os.path.exists(file_path):
    print("Downloading diamond dataset...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(file_path, "wb") as file:
        file.write(response.content)

df = pd.read_csv(file_path)
print("Original dataset shape:", df.shape)

# Step 2: Data cleaning
df = df.drop_duplicates().dropna().copy()

numeric_columns = [
    "carat", "depth", "table", "price", "x", "y", "z"
]

df = df[
    (df["carat"] > 0)
    & (df["price"] > 0)
    & (df["x"] > 0)
    & (df["y"] > 0)
    & (df["z"] > 0)
].copy()

df.to_csv("data/cleaned_diamonds.csv", index=False)
print("Cleaned dataset shape:", df.shape)

# Step 3: Visualizations
sns.set_theme(style="whitegrid")

# Pie chart: Cut distribution
cut_counts = df["cut"].value_counts()

plt.figure(figsize=(7, 6))
plt.pie(
    cut_counts.values,
    labels=cut_counts.index,
    autopct="%1.1f%%"
)
plt.title("Diamond Cut Distribution")
plt.tight_layout()
plt.savefig("reports/figures/cut_distribution.png")
plt.close()

# Bar chart: Color distribution
color_counts = df["color"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.bar(color_counts.index, color_counts.values)
plt.xlabel("Color")
plt.ylabel("Number of Diamonds")
plt.title("Diamond Color Distribution")
plt.tight_layout()
plt.savefig("reports/figures/color_distribution.png")
plt.close()

# Histogram: Price distribution
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="price", bins=25)
plt.title("Diamond Price Distribution")
plt.xlabel("Price (USD)")
plt.tight_layout()
plt.savefig("reports/figures/price_distribution.png")
plt.close()

# Histogram: Carat distribution
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="carat", bins=20)
plt.title("Diamond Carat Distribution")
plt.xlabel("Carat")
plt.tight_layout()
plt.savefig("reports/figures/carat_distribution.png")
plt.close()

print("Charts saved successfully!")

# Step 4: Machine learning
categorical = ["cut", "color", "clarity"]
numerical = ["carat", "depth", "table", "x", "y", "z"]
features = categorical + numerical

X = df[features]
y = df["price"]

preprocessor = ColumnTransformer([
    ("categories", OneHotEncoder(handle_unknown="ignore"),
     categorical),
    ("numbers", "passthrough", numerical)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

metrics = {
    "model": "Random Forest Regressor",
    "mean_absolute_error": round(float(mae), 2),
    "r2_score": round(float(r2), 4)
}

joblib.dump(model, "models/diamond_price_model.joblib")

with open("reports/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel evaluation:")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 4))
print("\nProject completed successfully!")
