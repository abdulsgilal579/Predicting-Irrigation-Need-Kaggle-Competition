import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── Load Data ──────────────────────────────────────────────
train = pd.read_csv("../Dataset/train.csv")
test  = pd.read_csv("../Dataset/test.csv")

print(f"Train Shape: {train.shape}")
print(f"Test Shape:  {test.shape}")
print(f"Train DTypes:\n{train.dtypes}")

# ── Target Distribution ────────────────────────────────────
print(f"\nValue Counts:\n{train['Irrigation_Need'].value_counts()}")
print(f"\nNormalized:\n{train['Irrigation_Need'].value_counts(normalize=True)}")

# ── Missing Values ─────────────────────────────────────────
print(f"\nMissing Values:\n{train.isnull().sum().sort_values(ascending=False)}")

# ── Categorical Columns ────────────────────────────────────
# automatically detect instead of hardcoding
categorical_cols = train.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('Irrigation_Need')

for col in categorical_cols:
    print(f"{col}: {train[col].nunique()} unique values → {train[col].unique()}")

# ── Encoding ───────────────────────────────────────────────
le_target = LabelEncoder()
train['Irrigation_Need_encoded'] = le_target.fit_transform(train['Irrigation_Need'])
print(f"\nTarget mapping: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

le = LabelEncoder()
for col in categorical_cols:
    train[col] = le.fit_transform(train[col])  # learn from train and convert
    test[col]  = le.transform(test[col])        # only convert, never re-learn

print("\nEncoding done!")

# ── Prepare Features and Target ────────────────────────────

X_train = train.drop(['id', 'Irrigation_Need', 'Irrigation_Need_encoded'], axis=1)
X_test  = test.drop(['id'], axis=1)
y_train = train['Irrigation_Need_encoded']

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")

print(train.head())