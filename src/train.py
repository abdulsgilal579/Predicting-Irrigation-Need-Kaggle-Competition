import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("../Dataset/train.csv")
test = pd.read_csv("../Dataset/test.csv")

# print(f"Train Shape: {train.shape}")
# print(f"Test Shape: {test.shape}")
# print(f"Train DTypes: {train.dtypes}")

# print(f"Train Irrigation Need Value Counts: {train['Irrigation_Need'].value_counts()}")
# print(f"Train Irrigation Need Value Counts Normalized: {train['Irrigation_Need'].value_counts(normalize=True)}")

# print(train.isnull().sum().sort_values(ascending=False))
categorical_cols = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage',
                    'Season', 'Irrigation_Type', 'Water_Source',
                    'Mulching_Used', 'Region']

for col in categorical_cols:
    print(f"{col}: {train[col].nunique()} unique values → {train[col].unique()}")
