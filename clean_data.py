import pandas as pd
import numpy as np

print("Loading and cleaning dataset...")

# 1. Load dataset
file_name = 'blood_cell_anomaly_detection.csv'
df = pd.read_csv(file_name)

print(f"Original dataset shape: {df.shape[0]} rows and {df.shape[1]} columns.")

# 2. Define target and unnecessary columns for cleaning
target_col = 'anomaly_label'

drop_cols = [
    'cell_id',                                  # Sequential identifier
    'disease_category',                         # Multi-class disease category
    'cell_type',                                # Detailed cell type
    'dataset_source',                           # Data leakage
    'staining_protocol',                        # Data leakage
    'cytodiffusion_anomaly_score',              # Score leakage
    'cytodiffusion_classification_confidence', # Confidence leakage
    'labeller_confidence_score'                 # Expert rating leakage
]

# Drop specified columns if present
cols_to_remove = [c for c in drop_cols if c in df.columns]
df_clean = df.drop(columns=cols_to_remove)

# 3. Handle missing values
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
for c in categorical_cols:
    df_clean[c] = df_clean[c].fillna(df_clean[c].mode()[0])

# 4. Apply One-Hot Encoding (get_dummies) for categorical columns (including microscope_model)
# We exclude the target column from encoding if it's already binary/numeric
cols_to_encode = [c for c in categorical_cols if c != target_col]
if cols_to_encode:
    df_clean = pd.get_dummies(df_clean, columns=cols_to_encode, drop_first=True, dtype=int)

# 5. Save cleaned dataset
df_clean.to_csv('cleaned_blood_cell_data.csv', index=False)
print(f"Data cleaned successfully! Current shape: {df_clean.shape[0]} rows and {df_clean.shape[1]} columns.")
print("Cleaned file saved as: cleaned_blood_cell_data.csv")