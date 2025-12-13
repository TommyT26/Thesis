import pandas as pd
import glob
import os
import utils

# --- ΡΥΘΜΙΣΕΙΣ ---
CLEAN_FOLDER = 'processed_data/'
BI_FOLDER = 'bidirectional_data/'

bi_files = glob.glob(os.path.join(BI_FOLDER, "bidirectional_*.csv"))
print(f"--- ADVANCED INTEGRITY CHECK ---")

# Βρίσκει τα αρχεία
bi_files = glob.glob(os.path.join(BI_FOLDER, "bidirectional_*.csv"))

print(f"--- ΕΝΑΡΞΗ ΕΛΕΓΧΟΥ ΑΚΕΡΑΙΟΤΗΤΑΣ ---")
print(f"Βρέθηκαν {len(bi_files)} αρχεία προς έλεγχο.\n")

for bi_path in bi_files:
    bi_filename = os.path.basename(bi_path)
    clean_filename = bi_filename.replace("bidirectional_", "")
    clean_path = os.path.join(CLEAN_FOLDER, clean_filename)
    
    if not os.path.exists(clean_path): continue

    print(f"Έλεγχος:{bi_filename}")

    # 1. Φόρτωση Δεδομένων
    # Διαβάζουμε όλο το Bi (είναι μικρό) και δείγμα του Clean (αν είναι τεράστιο)
    # Εδώ τα διαβάζουμε όλα για ακρίβεια. Αν έχεις λίγη RAM, βάλε nrows=100000
    df_bi = pd.read_csv(bi_path, low_memory=False)
    df_clean = pd.read_csv(clean_path, dtype=utils.DTYPES, low_memory=False)

    # Καθαρισμός Clean Bytes για να κάνουμε τη σούμα
    df_clean['Bytes_In_Num'] = df_clean['Bytes_In'].apply(utils.parse_mikrotik_bytes)
    df_clean['Bytes_Out_Num'] = df_clean['Bytes_Out'].apply(utils.parse_mikrotik_bytes)

    # Καθαρισμός Ports στο Clean
    df_clean['Src_Port'] = pd.to_numeric(df_clean['Src_Port'], errors='coerce').fillna(-1).astype(int)
    df_clean['Dst_Port'] = pd.to_numeric(df_clean['Dst_Port'], errors='coerce').fillna(-1).astype(int)

    # Spot Check με κατεύθυνση
    samples = df_bi.sample(n=min(5, len(df_bi)))

    for _, row in samples.iterrows():
        # Τα δεδομένα από το Bidirectional αρχείο
        ip_a, ip_b = row['IP_A'], row['IP_B']
        port_a, port_b = int(row['Port_A']), int(row['Port_B'])
        proto = row['Protocol']
        
        target_ab = row['Bytes_A_to_B']
        target_ba = row['Bytes_B_to_A']

        # Forward: Όταν Src == IP_A (άρα Dst == IP_B) και Ports αντίστοιχα
        mask_forward = (
            (df_clean['Src_IP'] == ip_a) & 
            (df_clean['Dst_IP'] == ip_b) &
            (df_clean['Src_Port'] == port_a) & 
            (df_clean['Dst_Port'] == port_b) &
            (df_clean['Protocol'] == proto)
        )

        # Backward: Όταν Src == IP_B (άρα Dst == IP_A) και Ports αντίστροφα!
        mask_backward = (
            (df_clean['Src_IP'] == ip_b) & 
            (df_clean['Dst_IP'] == ip_a) &
            (df_clean['Src_Port'] == port_b) &  
            (df_clean['Dst_Port'] == port_a) &
            (df_clean['Protocol'] == proto)
        )
        forward = df_clean[mask_forward]
        backward = df_clean[mask_backward]

        calc_ab = forward['Bytes_Out_Num'].sum() + backward['Bytes_In_Num'].sum()
        calc_ba = forward['Bytes_In_Num'].sum() + backward['Bytes_Out_Num'].sum()
        
        # Έλεγχος
        is_ab_ok = abs(calc_ab - target_ab) <= 1.0
        is_ba_ok = abs(calc_ba - target_ba) <= 1.0
        
        print(f"   Ζεύγος {ip_a}:{port_a} <-> {ip_b}:{port_b}")
        if is_ab_ok and is_ba_ok:
            print(f"         MATCH! A->B: {calc_ab} | B->A: {calc_ba}")
        else:
            print(f"         FAIL. Target A->B: {target_ab}, Found: {calc_ab}")
            print(f"               Target B->A: {target_ba}, Found: {calc_ba}")
            print(f"               Forward Rows: {len(forward)}, Backward Rows: {len(backward)}")

print("\nΤέλος ελέγχου.")
