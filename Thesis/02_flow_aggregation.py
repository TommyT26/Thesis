import pandas as pd
import glob
import os

INPUT_FOLDER = 'processed_data/'
OUTPUT_FOLDER = 'bidirectional_data/' 

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

clean_files = glob.glob(os.path.join(INPUT_FOLDER, 'clean_*.csv'))
print(f"Βρέθηκαν {len(clean_files)} καθαρά αρχεία για ενοποίηση.")

for file_path in clean_files:
    file_name = os.path.basename(file_path)
    print(f"Ενοποίηση flows στο αρχείο: {file_name}...")

    df = pd.read_csv(file_path, low_memory=False)

    print("  -> Καθαρισμός αριθμητικών πεδίων...")
    df['Bytes_In'] = pd.to_numeric(df['Bytes_In'], errors='coerce').fillna(0)
    df['Bytes_Out'] = pd.to_numeric(df['Bytes_Out'], errors='coerce').fillna(0)

    print("  -> Ταξινόμηση IP και Ports...")

    df['IP_A'] = df.apply(lambda x: min(str(x['Src_IP']), str(x['Dst_IP'])), axis=1)
    df['IP_B'] = df.apply(lambda x: max(str(x['Src_IP']), str(x['Dst_IP'])), axis=1)

    df['Port_A'] = df.apply(lambda x: min(x['Src_Port'], x['Dst_Port']), axis=1)
    df['Port_B'] = df.apply(lambda x: max(x['Src_Port'], x['Dst_Port']), axis=1)

    df['Total_Bytes_Row'] = df['Bytes_In'] + df['Bytes_Out']

    print("  -> Ομαδοποίηση (Grouping)...")

    grouped = df.groupby(['IP_A', 'IP_B', 'Port_A', 'Port_B', 'Protocol']).agg({
        'Total_Bytes_Row': 'sum',
        'Event': 'count'
    }).reset_index()

    grouped.rename(columns={
        'Total_Bytes_Row': 'Total_Volume_Bytes',
        'Event': 'Flow_Count'
    }, inplace=True)

    save_path = os.path.join(OUTPUT_FOLDER, "bi_" + file_name)
    grouped.to_csv(save_path, index=False)

    print(f"--> Έτοιμο! Από {len(df)} γραμμές, μειώθηκε σε {len(grouped)} μοναδικές συνομιλίες")

print("\nΗ διαδικασία ολοκληρώθηκε. Τα αρχεία στον φάκελο 'bidirectional_data' είναι έτοιμα!")