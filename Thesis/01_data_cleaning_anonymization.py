import pandas as pd
import hashlib
import os
import glob

#--------ρυθμίσεις φακέλων-----------
INPUT_FOLDER = 'raw_data/'
OUTPUT_FILE = 'processed_data/'

if not os.path.exists(OUTPUT_FILE):
    os.makedirs(OUTPUT_FILE)

COLUMN_NAMES = [
    "Event",
    "XEvent",
    "Protocol",
    "Src_IP",
    "Src_Port",
    "Dst_IP",
    "Dst_Port",
    "XSrc_IP",
    "XSrc_Port",
    "XDst_IP",
    "XDst_Port",
    "Bytes_In",
    "Bytes_Out",
]

def annonymize_ip(ip):
    if pd.isna(ip) or str(ip).strip() in ["", "-"]:
        return "unknown"
    raw_str = str(ip) + SALT_KEY
    return hashlib.sha256(raw_str.encode()).hexdigest()[:16]

csv_files = glob.glob(os.path.join(INPUT_FOLDER, '*.csv'))
print(f"Βρέθηκαν {len(csv_files)} αρχεία.")

for file_path in csv_files:
    file_name = os.path.basename(file_path)
    print(f"Επεξεργασία αρχείου: {file_name}")

    chunk_size = 500000
    chunks = pd.read_csv(file_path, chunksize=chunk_size, header=None, names=COLUMN_NAMES, encoding='utf-8', on_bad_lines='skip', low_memory=False)
    
    header_written = False

    for chunk in chunks:
        ip_cols_to_hash = ["Src_IP", "Dst_IP", "XSrc_IP", "XDst_IP"]

        for col in ip_cols_to_hash:
            if col in chunk.columns:
                chunk[col] = chunk[col].apply(annonymize_ip)
        
        save_path = os.path.join(OUTPUT_FILE, "clean_" + file_name)
        mode = 'w' if not header_written else 'a'
        chunk.to_csv(save_path, index=False, header=(not header_written), mode=mode)
        header_written = True

    print(f"--> Έτοιμο: clean_{file_name}")

print("\nΌλα τα αρχεία στον φάκελο 'processed_data' επεξεργάστηκαν επιτυχώς.")