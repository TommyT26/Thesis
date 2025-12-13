import pandas as pd
import hashlib
import hmac
import os
import glob
import psutil
from dotenv import load_dotenv

#*---- 1.ΡΥΘΜΙΣΕΙΣ ----
# Φορτώνουμε το κλειδί από το αρχείο .env
load_dotenv()
SALT_KEY_STR = os.getenv("THESIS_SALT_KEY")

# Έλεγχος αν λείπει το κλειδί
if not SALT_KEY_STR:
    raise ValueError("ΣΦΑΛΜΑ: Δεν βρέθηκε το κλειδί στο αρχείο .env")

# Μετατροπή του κλειδιού σε bytes για χρήση με HMAC
SALT_KEY_BYTES = SALT_KEY_STR.encode('utf-8')

# Φάκελοι ειδόδου και εξόδου
INPUT_FOLDER = 'raw_data/'
OUTPUT_FOLDER = 'processed_data/'

# Δημιουργία φακέλου εξόδου αν δεν υπάρχει
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

#*---- 2.ΟΡΙΣΜΟΣ ΣΤΗΛΩΝ ---- 
# Τα raw αρχεία του MikroTik δεν έχουν κεφαλίδες
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

#*---- 3.ΣΥΝΑΡΤΙΣΗ ΥΠΟΛΟΓΙΣΜΟΥ ΒΕΛΤΙΣΤΟΥ ΜΕΓΕΘΟΥΣ CHUNK ----
# Υπολογίζει το βέλτιστο μέγεθος chunk βάσει της διαθέσιμης μνήμης.
def get_optimal_chunk_size(file_path, safety_factor=0.15):
    try:
        mem = psutil.virtual_memory()
        available_ram = mem.available
        sample = pd.read_csv(
            file_path, 
            nrows=2000,
            header=None,
            names=COLUMN_NAMES,
            low_memory=False)
        sample_memory_bytes = sample.memory_usage(deep=True).sum()
        bytes_per_row = sample_memory_bytes / 2000

        target_chunk_memory = available_ram * safety_factor
        optimal_size = int(target_chunk_memory / bytes_per_row)

        return max(10000, min(optimal_size, 2000000))
    except Exception:
        return 500000

#*---- 4.ΣΥΝΑΡΤΗΣΗ ΑΝΩΝΥΜΟΠΟΙΗΣΗΣ IP ----
# Δέχεται μια IP και επιστρέφει ένα HMAC-SHA256 hash 16 χαρακτήρων 
def anonymize_ip(ip):
    # Αν η IP είναι κενή ή παύλα, επιστρέφει "unknown"
    if pd.isna(ip) or str(ip).strip() in ["", "-"]:
        return "unknown"
    
    # Μετατροπή της IP σε bytes
    msg_bytes = str(ip).encode('utf-8')

    # Δημιουργία HMAC (Κλειδί, Μήνυμα, Αλγόριθμος)
    hmac_obj = hmac.new(SALT_KEY_BYTES, msg_bytes, hashlib.sha256)

    # Επιστρέφει τους πρώτους 16 χαρακτήρες του hash
    return hmac_obj.hexdigest()[:16]

#*---- 5.ΕΠΕΞΕΡΓΑΣΙΑ ΑΡΧΕΙΩΝ ----
# Αναζήτηση αρχείων προς επεξεργασία και εκτύπωση πλήθους
csv_files = glob.glob(os.path.join(INPUT_FOLDER, '*.csv'))
print(f"Βρέθηκαν {len(csv_files)} αρχεία.")

for file_path in csv_files:
    file_name = os.path.basename(file_path)
    print(f"Επεξεργασία αρχείου: {file_name}")

    # Υπολογισμός βέλτιστου μεγέθους chunk
    current_chunk_size = get_optimal_chunk_size(file_path)
    print(f"   --> Chunk Size: {current_chunk_size:,} γραμμές")

    chunks = pd.read_csv(
        file_path,
        chunksize=current_chunk_size, # Χρήση του υπολογισμένου μεγέθους chunk
        header=None,            # Δεν έχει κεφαλίδες
        names=COLUMN_NAMES,     # Βάζει τις κεφαλίδες
        encoding='utf-8', 
        on_bad_lines='skip',    # Προσπερνάει χαλασμένες γραμμές
        low_memory=False
    )
    
    header_written = False

    for i, chunk in enumerate(chunks):
        # Εντοπίζει τις στήλες IP για ανωνυμοποίηση
        ip_cols_to_hash = ["Src_IP", "Dst_IP", "XSrc_IP", "XDst_IP"]

        # Εφαρμογή της ανωνυμοποίησης σε κάθε στήλη IP
        for col in ip_cols_to_hash:
            if col in chunk.columns:
                chunk[col] = chunk[col].apply(anonymize_ip)

        # Αποθηκεύει το επεξεργασμένο κομμάτι στο νέο αρχείο εξόδου
        save_path = os.path.join(OUTPUT_FOLDER, "clean_" + file_name)

        # Τσεκάρει αν πρέπει να γράψει την κεφαλίδα ή να προσθέσει χωρίς αυτήν
        mode = 'w' if not header_written else 'a'
        chunk.to_csv(save_path, index=False, header=(not header_written), mode=mode)
        header_written = True

    print(f"--> Έτοιμο: clean_{file_name}")

print("\nΌλα τα αρχεία στον φάκελο 'processed_data' επεξεργάστηκαν επιτυχώς.")
