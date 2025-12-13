import pandas as pd
import glob
import os
import numpy as np

INPUT_FOLDER = 'processed_data/'
OUTPUT_FOLDER = 'bidirectional_data/'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

clean_files = glob.glob(os.path.join(INPUT_FOLDER, 'clean_*.csv'))
print(f"Βρέθηκαν {len(clean_files)} καθαρά αρχεία.")

for file_path in clean_files:
    file_name = os.path.basename(file_path)
    print(f"Ενοποίηση flows στο αρχείο: {file_name}...")

    df = pd.read_csv(file_path, low_memory=False)

    df['Bytes_In'] = pd.to_numeric(df['Bytes_In'], errors='coerce').fillna(0)
    df['Bytes_Out'] = pd.to_numeric(df['Bytes_Out'], errors='coerce').fillna(0)
    
    df['Flow_Bytes'] = df['Bytes_In'] + df['Bytes_Out']

    df['Src_IP'] = df['Src_IP'].astype(str)
    df['Dst_IP'] = df['Dst_IP'].astype(str)

    df['IP_A'] = df.apply(lambda x: min(x['Src_IP'], x['Dst_IP']), axis=1)
    df['IP_B'] = df.apply(lambda x: max(x['Src_IP'], x['Dst_IP']), axis=1)

    df['Port_A'] = df.apply(lambda x: min(int(float(x['Src_Port'])), int(float(x['Dst_Port']))), axis=1)
    df['Port_B'] = df.apply(lambda x: max(int(float(x['Src_Port'])), int(float(x['Dst_Port']))), axis=1)

    print("  -> Διαχωρισμός κατεύθυνσης (Upload/Download)...")
    
    df['Bytes_A_to_B'] = np.where(df['Src_IP'] == df['IP_A'], df['Flow_Bytes'], 0)
    df['Bytes_B_to_A'] = np.where(df['Src_IP'] == df['IP_B'], df['Flow_Bytes'], 0)

    print("  -> Ομαδοποίηση...")

    grouped = df.groupby(['IP_A', 'IP_B', 'Port_A', 'Port_B', 'Protocol']).agg({
        'Bytes_A_to_B': 'sum',   
        'Bytes_B_to_A': 'sum',  
        'Event': 'count'
    }).reset_index()

    grouped.rename(columns={'Event': 'Flow_Count'}, inplace=True)

    grouped['Total_Bytes'] = grouped['Bytes_A_to_B'] + grouped['Bytes_B_to_A']

    save_path = os.path.join(OUTPUT_FOLDER, "bi_" + file_name)
    grouped.to_csv(save_path, index=False)
    
    print(f"--> Έτοιμο: {file_name}")

print("\nΗ ενοποίηση ολοκληρώθηκε με διαχωρισμό Upload/Download!")