import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocessing(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)

    # ===== Column grouping =====
    binary_cols = [
        col for col in df.columns
        if set(df[col].dropna().unique()).issubset({0, 1})
    ]

    numeric_cols = df.select_dtypes(include=['number']).columns.difference(binary_cols)
    categorical_cols = df.select_dtypes(include=['object']).columns

    # ===== Missing value handling =====
    df['insurance_type'] = df['insurance_type'].fillna('Unknown')

    # ===== Outlier handling =====
    df['annual_medical_cost'] = np.log1p(df['annual_medical_cost'])
    df['bmi'] = df['bmi'].clip(lower=15)

    # ===== Binary encoding =====
    df['smoker'] = df['smoker'].map({'No': 0, 'Yes': 1})

    # ===== Ordinal encoding =====
    df['physical_activity_level'] = df['physical_activity_level'].map({
        'Low': 1, 'Medium': 2, 'High': 3
    })

    df['city_type'] = df['city_type'].map({
        'Rural': 1, 'Semi-Urban': 2, 'Urban': 3
    })

    # ===== One-hot encoding =====
    df = pd.get_dummies(
        df,
        columns=['gender', 'insurance_type'],
        prefix=['gender', 'insurance'],
        drop_first=False
    )

    # Convert bool → int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Drop redundant columns
    df.drop(columns=['gender_Male', 'insurance_Unknown'], inplace=True)

    binary_cols2 = [col for col in numeric_cols if set(df[col].dropna().unique()).issubset({0, 1})]
    cols_to_standardize = [col for col in numeric_cols if col not in binary_cols2]
    
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[cols_to_standardize] = scaler.fit_transform(df[cols_to_standardize])
    df_scaled = pd.DataFrame(df_scaled,columns = df.columns)

    # ===== Save processed data =====
    df_scaled.to_csv(output_path, index=False)

    return df_scaled

if __name__ == "__main__":
    preprocessing(
        input_path="medical_cost_prediction_dataset.csv",
        output_path="preprocessing/medical_cost_preprocessed.csv",
    )