# src/preprocessing/clean_anonymize.py
import os
import glob
import pandas as pd

from src.config import PATHS, CHUNK_SIZE, INTERNAL_SUBNET, SUBNET_PREFIXES
from src.utils.memory import process_in_chunks
from src.utils.anonymize import anonymize_series

# Το raw export του MikroTik έχει 14 πεδία. Τα δύο τελευταία είναι In Byte
# και Out Byte, όπως και στον πίνακα του proposal· στήλη packets δεν υπάρχει.
# Το Out Byte είναι 0 σε όλες τις 25,4 εκατ. εγγραφές, οπότε το κρατώ μόνο
# για πιστότητα και δεν το χρησιμοποιώ πουθενά.
MIKROTIK_COLUMNS = [
    'Time', 'Status', 'Action', 'Protocol',
    'Src_IP', 'Src_Port', 'Dst_IP', 'Dst_Port',
    'NAT_Src_IP', 'NAT_Src_Port', 'NAT_Dst_IP', 'NAT_Dst_Port',
    'Bytes', 'Out_Byte'
]

# Όλες οι στήλες με IP ανωνυμοποιούνται, και οι NAT: αλλιώς διαρρέουν πραγματικές IP
IP_COLUMNS = ['Src_IP', 'Dst_IP', 'NAT_Src_IP', 'NAT_Dst_IP']

# Μετρητές για έλεγχο ορθότητας της NAT επίλυσης (μηδενίζονται σε κάθε αρχείο)
STATS = {'rows': 0, 'src_resolved': 0, 'dst_resolved': 0}


def resolve_host(orig, nat):
    """
    Επιλύει το NAT: αν η IP μετά το NAT (X-Src/X-Dst) είναι εσωτερική ενώ η αρχική δεν είναι,
    ο πραγματικός host είναι η εσωτερική. Παράδειγμα (εισερχόμενη κίνηση):
        Dst_IP = δημόσια IP του router, NAT_Dst_IP = 192.168.7.90  ->  Dst_Host = 192.168.7.90
    Σε εξερχόμενη κίνηση η Src_IP είναι ήδη εσωτερική και μένει ως έχει.
    """
    # Όλα τα δικά μας υποδίκτυα, όχι μόνο το INTERNAL_SUBNET. Με ένα μόνο
    # πρόθεμα, ένα ζεύγος NAT που έδειχνε σε 192.168.8.x ή 192.168.1.x δεν
    # επιλυόταν και η ροή χρεωνόταν στη δημόσια διεύθυνση του router.
    # Μετρημένο στα καθαρισμένα δεδομένα: 66.598 γραμμές από 25.403.323
    # (0,26%) αλλάζουν host με αυτή τη διόρθωση. Μικρό ποσοστό, αλλά απαιτεί
    # ξανατρέξιμο από το στάδιο clean για να εφαρμοστεί.
    prefixes = tuple(SUBNET_PREFIXES) or (INTERNAL_SUBNET,)
    nat_internal = nat.str.startswith(prefixes, na=False)
    orig_internal = orig.str.startswith(prefixes, na=False)
    use_nat = nat_internal & ~orig_internal
    return orig.mask(use_nat, nat), use_nat


def process_raw_chunk(df_chunk):
    """Καθαρίζει ένα chunk τη φορά."""
    df_chunk = df_chunk.copy()

    # Καθαρισμός κενών χαρακτήρων (τα κενά strings γίνονται NA)
    for col in IP_COLUMNS:
        s = df_chunk[col].astype('string').str.strip()
        df_chunk[col] = s.mask(s == '')

    # Αφαίρεση γραμμών χωρίς πηγή/προορισμό
    df_chunk = df_chunk.dropna(subset=['Src_IP', 'Dst_IP'])

    # NAT επίλυση: οι στήλες Src_Host/Dst_Host δείχνουν τον πραγματικό (εσωτερικό αν υπάρχει) host
    df_chunk['Src_Host'], src_nat = resolve_host(df_chunk['Src_IP'], df_chunk['NAT_Src_IP'])
    df_chunk['Dst_Host'], dst_nat = resolve_host(df_chunk['Dst_IP'], df_chunk['NAT_Dst_IP'])
    STATS['rows'] += len(df_chunk)
    STATS['src_resolved'] += int(src_nat.sum())
    STATS['dst_resolved'] += int(dst_nat.sum())

    # Ανωνυμοποίηση
    for col in IP_COLUMNS + ['Src_Host', 'Dst_Host']:
        df_chunk[col] = anonymize_series(df_chunk[col])

    return df_chunk


def run():
    print("\n--- Έναρξη: Καθαρισμός και Ανωνυμοποίηση (με Chunking & Progress) ---")

    raw_files = glob.glob(os.path.join(PATHS['raw_data'], "*.csv"))

    if not raw_files:
        print(f"Προειδοποίηση: Δεν βρέθηκαν .csv αρχεία στον φάκελο {PATHS['raw_data']}")
        return

    total_files = len(raw_files)

    for index, file_path in enumerate(raw_files, start=1):
        filename = os.path.basename(file_path)
        print(f"\nΕπεξεργασία αρχείου ({index}/{total_files}): {filename}")

        output_path = os.path.join(PATHS['processed_data'], f"clean_{filename}")
        STATS.update(rows=0, src_resolved=0, dst_resolved=0)

        # Κάθε chunk γράφεται κατευθείαν στο αρχείο εξόδου (χωρίς pd.concat όλων στη μνήμη)
        rows = process_in_chunks(
            file_path,
            chunk_size=CHUNK_SIZE,
            process_function=process_raw_chunk,
            output_path=output_path,
            current_file=index,
            total_files=total_files,
            header=None,
            names=MIKROTIK_COLUMNS
        )
        print(f" -> Αποθηκεύτηκαν {rows} γραμμές στο: {output_path}")
        print(f" -> NAT επίλυση: Src_Host από NAT = {STATS['src_resolved']} | "
              f"Dst_Host από NAT = {STATS['dst_resolved']} (από {STATS['rows']} γραμμές)")

    print("--- Ολοκληρώθηκε ---")
