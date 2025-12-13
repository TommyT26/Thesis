import pandas as pd
import glob
import os
import numpy as np
import utils  # Το αρχείο utils.py που φτιάξαμε
import gc
from scipy.stats import entropy

# *---- 1. ΡΥΘΜΙΣΕΙΣ ----
INPUT_FOLDER = 'bidirectional_data/'
OUTPUT_FOLDER = 'final_profiles/'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# *---- 2. HELPER FUNCTIONS ----
# Υπολογίζει το Shannon Entropy μιας σειράς (π.χ. Ports).
# High Entropy = Random Port Scanning / Malware
#Low Entropy = Specific Service (e.g. Web Server always port 80)   
def calculate_entropy(series):
    counts = series.value_counts()
    return entropy(counts)

# Διαβάζει το bidirectional αρχείο και το "σπάει" σε Host View (Option A).
def process_file_to_host_view(file_path):
    # Διαβάζουμε μόνο τα απαραίτητα
    cols = [
    "IP_A", "IP_B",
    "Port_A", "Port_B",
    "Bytes_A_to_B", "Bytes_B_to_A",
    "Flow_Count", "Protocol"
]
    df = pd.read_csv(file_path, usecols=cols, low_memory=False)
    
    # --- VIEW A: Ο Host είναι η IP_A ---
    # Ο A μιλάει στον B (Peer). Το Dst_Port είναι το Port του B.
    df_a = df.rename(columns={
        "IP_A": "Host",
        "IP_B": "Peer",
        "Port_B": "Dst_Port",
        "Bytes_A_to_B": "Bytes_Out", # A στέλνει (Outbound)
        "Bytes_B_to_A": "Bytes_In"   # A λαμβάνει (Inbound)
    })
    
    # --- VIEW B: Ο Host είναι η IP_B ---
    # Ο B μιλάει στον A (Peer). Το Dst_Port είναι το Port του A.
    df_b = df.rename(columns={
        "IP_B": "Host",
        "IP_A": "Peer",
        "Port_A": "Dst_Port",  # ΠΡΟΣΟΧΗ: Εδώ το Dst Port είναι το Port_A
        "Bytes_B_to_A": "Bytes_Out", # B στέλνει (Outbound)
        "Bytes_A_to_B": "Bytes_In"   # B λαμβάνει (Inbound)
    })
    
    # Κρατάμε μόνο τις κοινές στήλες
    common_cols = ["Host", "Peer", "Dst_Port", "Bytes_Out", "Bytes_In", "Flow_Count", "Protocol"]
    
    return pd.concat([df_a[common_cols], df_b[common_cols]], axis=0, ignore_index=True)

# *---- 3. ΚΥΡΙΑ ΔΙΑΔΙΚΑΣΙΑ ----
files = glob.glob(os.path.join(INPUT_FOLDER, "bidirectional_*.csv"))
print(f"--- HOST PROFILING (Step 04) ---")
print(f"Βρέθηκαν {len(files)} αρχεία.")

# Λίστα για να μαζέψουμε τα δεδομένα από όλα τα αρχεία
all_data = []

# Φάση 1: Φόρτωση και Stacking
for file_path in files:
    print(f"Φόρτωση και αναδιάταξη: {os.path.basename(file_path)}")
    host_view_df = process_file_to_host_view(file_path)
    all_data.append(host_view_df)

if not all_data:
    raise ValueError("Δεν βρέθηκαν δεδομένα!")

print("Συγχώνευση όλων των δεδομένων στη μνήμη...")
master_df = pd.concat(all_data, ignore_index=True)

# Καθαρισμός για μνήμη
del all_data
gc.collect()

print(f"Σύνολο εγγραφών (Flows): {len(master_df):,}")

# *---- 4. AGGREGATION (CORE STEP) ----
print("Υπολογισμός Features ανά Host...")

# One-hot encoding για τα πρωτόκολλα
master_df['Is_TCP'] = (master_df['Protocol'].str.lower() == 'tcp').astype(int)
master_df['Is_UDP'] = (master_df['Protocol'].str.lower() == 'udp').astype(int)

# Group By Host
profiles = master_df.groupby("Host").agg(
    # 4.1 Volume Features
    Total_Bytes_Out=("Bytes_Out", "sum"),
    Total_Bytes_In=("Bytes_In", "sum"),
    Total_Flows=("Flow_Count", "sum"),
    
    # 4.3 Διαφορετικοί Peers
    Unique_Peers=("Peer", "nunique"),
    
    # 4.4 Συνολικά Διαφορετικά Ports
    Unique_Dst_Ports=("Dst_Port", "nunique"),
    # Port Entropy (Πιο αργό, αλλά πολύτιμο για Scanners)
    Port_Entropy=("Dst_Port", calculate_entropy),
    
    # 4.5 Protocol Mix
    TCP_Flows=("Is_TCP", "sum"),
    UDP_Flows=("Is_UDP", "sum")
)

# Clean up memory
del master_df
gc.collect()

# *---- 5. DERIVED FEATURES (VECTORIZATION) ----
print("Υπολογισμός Παραγόμενων Δεικτών (Ratios & Logs)...")

# 4.1 Volume Sum
profiles["Total_Bytes"] = profiles["Total_Bytes_In"] + profiles["Total_Bytes_Out"]

# 4.2 Directionality Features
# Out_In_Ratio = Out / (In + 1)
profiles["Out_In_Ratio"] = profiles["Total_Bytes_Out"] / (profiles["Total_Bytes_In"] + 1)

# Asymmetry = |Out - In| / (Out + In + 1)
profiles["Asymmetry"] = abs(profiles["Total_Bytes_Out"] - profiles["Total_Bytes_In"]) / (profiles["Total_Bytes"] + 1)

# 4.3 Peer Diversity Ratio
# Flows per Peer (Scanning indicator if Peers is high and Flows/Peer is low)
profiles["Flows_per_Peer"] = profiles["Total_Flows"] / (profiles["Unique_Peers"] + 1)

# 4.5 Protocol Ratios
profiles["TCP_Ratio"] = profiles["TCP_Flows"] / (profiles["Total_Flows"] + 1)
profiles["UDP_Ratio"] = profiles["UDP_Flows"] / (profiles["Total_Flows"] + 1)

# *---- 6. PREPARE FOR PLOTTING (LOGS) ----
# Log scaling is essential for clustering visualisations later
profiles["Log_Total_Bytes"] = np.log1p(profiles["Total_Bytes"])
profiles["Log_Total_Flows"] = np.log1p(profiles["Total_Flows"])
profiles["Log_Unique_Peers"] = np.log1p(profiles["Unique_Peers"])

# *---- 7. FINAL CLEANUP & SAVE ----
profiles = profiles.reset_index()
profiles = profiles.sort_values("Total_Bytes", ascending=False)

output_file = os.path.join(OUTPUT_FOLDER, "host_behavioral_vectors.csv")
profiles.to_csv(output_file, index=False)

print(f"\nΕΤΟΙΜΟ! Αποθηκεύτηκε στο: {output_file}")
print(f"   Πλήθος Hosts: {len(profiles)}")
print("\n--- Δείγμα Δεδομένων (Top 5) ---")
print(profiles[["Host", "Log_Total_Bytes", "Out_In_Ratio", "Port_Entropy", "Unique_Peers"]].head())