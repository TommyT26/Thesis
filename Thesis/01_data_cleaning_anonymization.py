import pandas as pd
import hashlib
import hmac
import os
import glob
from dotenv import load_dotenv
import utils

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

# *---- 2. ΣΥΝΑΡΤΗΣΗ HMAC (Core Logic) ----
def compute_hmac(ip_str):
    #Check αν το ip είναι NaN ή κενό
    s = str(ip_str).strip()
    if s in ['', '-', 'nan', 'NaN', 'None']:
        return 'unknown'
    
    # Υπολογίζει το HMAC για ένα string.
    msg_bytes = s.encode('utf-8')
    return hmac.new(SALT_KEY_BYTES, msg_bytes, hashlib.sha256).hexdigest()[:16]

# *---- 3. ΕΠΕΞΕΡΓΑΣΙΑ ΑΡΧΕΙΩΝ (VECTORIZED STYLE) ----
# Αναζήτηση αρχείων προς επεξεργασία και εκτύπωση πλήθους
csv_files = glob.glob(os.path.join(INPUT_FOLDER, '*.csv'))
print(f"Βρέθηκαν {len(csv_files)} αρχεία.")

for file_path in csv_files:
    file_name = os.path.basename(file_path)
    print(f"Επεξεργασία αρχείου: {file_name}")

    # Υπολογισμός βέλτιστου μεγέθους chunk
    current_chunk_size = utils.get_optimal_chunk_size(file_path, column_names=utils.COLUMN_NAMES)
    print(f"   --> Chunk Size: {current_chunk_size:,} γραμμές")

    chunks = pd.read_csv(
        file_path,
        chunksize=current_chunk_size, # Χρήση του υπολογισμένου μεγέθους chunk
        header=None,            # Δεν έχει κεφαλίδες
        names=utils.COLUMN_NAMES,     # Βάζει τις κεφαλίδες
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
                # --- ΒΕΛΤΙΣΤΟΠΟΙΗΣΗ (Vector-Style) ---
                
                # 1. Βρίσκει τις μοναδικές IPs σε αυτό το chunk (αφαιρώντας τα NaN)
                unique_ips = chunk[col].dropna().unique()

                # 2. Φτιάχνει ένα λεξικό { '192.168.1.1': 'hash_xyz', ... }
                # Υπολογίζουμε το HMAC μόνο μία φορά για κάθε μοναδική IP!
                ip_map = {ip: compute_hmac(str(ip)) for ip in unique_ips}

                # 3. Αντικατάσταση (Map) - Πολύ γρήγορο σε Pandas
                # Όπου δεν βρει IP (π.χ. NaN), βάζει 'unknown'
                chunk[col] = chunk[col].map(ip_map).fillna('unknown')

        # Αποθηκεύει το επεξεργασμένο κομμάτι στο νέο αρχείο εξόδου
        save_path = os.path.join(OUTPUT_FOLDER, "clean_" + file_name)

        # Τσεκάρει αν πρέπει να γράψει την κεφαλίδα ή να προσθέσει χωρίς αυτήν
        mode = 'w' if not header_written else 'a'
        chunk.to_csv(save_path, index=False, header=(not header_written), mode=mode)
        header_written = True

    print(f"--> Έτοιμο: clean_{file_name}")

print("\nΌλα τα αρχεία στον φάκελο 'processed_data' επεξεργάστηκαν επιτυχώς.")
