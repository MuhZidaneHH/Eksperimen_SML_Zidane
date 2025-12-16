import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# --- KONFIGURASI PATH ---
# Menggunakan relative path agar jalan di Local maupun GitHub Actions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, '..', 'TelcoCustomerChurn_raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'TelcoCustomerChurn_preprocessing')

def process_data():
    print(f"Mencari data di: {RAW_DATA_PATH}")
    
    # 1. Load Data
    try:
        df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print("Error: File dataset tidak ditemukan!")
        return

    # 2. Preprocessing Steps
    
    # Drop CustomerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Fix TotalCharges (String to Numeric)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Encoding Binary & Categorical
    # Label Encoding untuk kolom biner sederhana
    le = LabelEncoder()
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
            
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
        
    # One-Hot Encoding sisa kategori
    df = pd.get_dummies(df, drop_first=True)
    
    # 3. Split Data
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Scaling
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    valid_num_cols = [c for c in num_cols if c in X_train.columns]
    
    X_train[valid_num_cols] = scaler.fit_transform(X_train[valid_num_cols])
    X_test[valid_num_cols] = scaler.transform(X_test[valid_num_cols])
    
    # 5. Save Data
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_train.to_csv(os.path.join(OUTPUT_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, 'y_test.csv'), index=False)
    
    print(f"SUKSES! Data hasil preprocessing tersimpan di: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_data()