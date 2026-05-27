import pandas as pd
import os

# Define the path to our new dataset
dataset_path = os.path.join('datasets', 'voice', 'current', 'pd_speech_features.csv')

def load_and_explore_data(filepath):
    """
    Loads a dataset from a CSV file and prints basic exploratory information.
    """
    print(f"--- Loading data from {filepath} ---")
    
    try:
        # 1. Load the dataset using pandas
        # pd.read_csv() reads a comma-separated values (csv) file into DataFrame
        # We use header=1 because the dataset has an extra metadata row at the top
        df = pd.read_csv(filepath, header=1)
        print("\n[SUCCESS] Dataset loaded successfully!\n")
        
        # 2. Print the shape of the dataset
        print("--- Dataset Shape ---")
        print(f"Total Rows (Samples): {df.shape[0]}")
        print(f"Total Columns (Features): {df.shape[1]}\n")
        
        # 3. Print the column names
        print("--- Column Names ---")
        print(df.columns.tolist())
        print("\n")
        
        # 4. Automatically identify target column
        target_candidates = ['class', 'status', 'diagnosis']
        target_column = None
        for col in df.columns:
            if col.lower() in target_candidates:
                target_column = col
                break
                
        if target_column:
            print(f"[SUCCESS] Detected target column: '{target_column}'\n")
        else:
            print("[ERROR] Could not automatically detect a target column.\n")
        
        # 5. Print the first 5 rows
        print("--- First 5 Rows ---")
        print(df.head())
        
    except FileNotFoundError:
        print(f"[ERROR] The file at {filepath} was not found.")
        print("Please make sure you have placed 'pd_speech_features.csv' inside 'datasets/voice/current/'.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    load_and_explore_data(dataset_path)
