import pandas as pd
import glob
import os
import numpy as np
import utils
import gc     # Garbage Collector για διαχείριση μνήμης

#*---- 1.ΡΥΘΜΙΣΕΙΣ ----

# Φάκελοι εισόδου και εξόδου
INPUT_FOLDER = 'processed_data/'
OUTPUT_FOLDER = 'bidirectional_data/' 
               
# Δημιουργία φακέλου εξόδου αν δεν υπάρχει
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Εύρεση όλων των καθαρών αρχείων
files = glob.glob(os.path.join(INPUT_FOLDER, "clean_*.csv"))
print(f"Βρέθηκαν {len(files)} καθαρά αρχεία.")

#*---- 2.ΕΠΕΞΕΡΓΑΣΙΑ ΑΡΧΕΙΩΝ ----

for file_path in files:
    file_name = os.path.basename(file_path)
    print(f"\nΕπεξεργασία αρχείου: {file_name}")

    # Αυτόματος υπολογισμός Chunk Size
    current_chunk_size = utils.get_optimal_chunk_size(file_path)
    print(f"   --> Chunk Size: {current_chunk_size:,} γραμμές")

    # Δημιουργία λίστας για την αποθήκευση των επεξεργασμένων chunks
    aggregated_chunks = []

    # low_memory=False για την αποφυγή προειδοποιήσεων τύπου
    reader = pd.read_csv(
        file_path,
        dtype=utils.DTYPES,
        chunksize=current_chunk_size,
        low_memory=False,
        on_bad_lines='skip'
    )

    # Επεξεργασία κάθε chunk
    for i, chunk in enumerate(reader, start=1):

        # Βελτιστοποίηση: Map αντί για apply σε κάθε γραμμή
        for col in ["Bytes_In", "Bytes_Out"]:
            unique_vals = chunk[col].unique()
            val_map = {val: utils.parse_mikrotik_bytes(val) for val in unique_vals}
            chunk[col] = chunk[col].map(val_map).fillna(0.0)

        # Μετατροπή Ports σε Int.
        # ΣΗΜΕΙΩΣΗ: Το -1 χρησιμοποιείται για "Unknown/Invalid Port".
        chunk["Src_Port"] = pd.to_numeric(chunk["Src_Port"], errors='coerce').fillna(-1).astype(int)
        chunk["Dst_Port"] = pd.to_numeric(chunk["Dst_Port"], errors='coerce').fillna(-1).astype(int)

        # Πετάει τις γραμμές που δεν έχουν κίνηση
        chunk = chunk[(chunk["Bytes_In"] + chunk["Bytes_Out"]) > 0].copy()

        if len(chunk) == 0: continue

        #! ---------------------------------------------------------
        #! ΣΧΟΛΙΟ ΑΣΦΑΛΕΙΑΣ (FEEDBACK UPDATE):
        #! Η σύγκριση εδώ είναι ΛΕΞΙΚΟΓΡΑΦΙΚΗ (αλφαβητική) πάνω στα Hashes.
        #! Το 'IP_A' είναι απλά το hash που προηγείται αλφαβητικά.
        #! ΠΡΟΣΟΧΗ: Το IP_A ΔΕΝ σημαίνει απαραίτητα Client/Source.
        #! Το IP_B ΔΕΝ σημαίνει απαραίτητα Server/Destination.
        #! Χρησιμοποιείται ΜΟΝΟ για να ομαδοποιηθεί η κίνηση A<->B και B<->A στο ίδιο κλειδί.
        #! ---------------------------------------------------------
        # Βρίσκει ποια είναι η μικρότερη IP και ποια η μεγαλύτερη
        chunk["IP_A"] = np.where(chunk["Src_IP"] < chunk["Dst_IP"], chunk["Src_IP"], chunk["Dst_IP"])
        chunk["IP_B"] = np.where(chunk["Src_IP"] < chunk["Dst_IP"], chunk["Dst_IP"], chunk["Src_IP"])

        # Δημιουργεί μάσκες για να εντοπίσει τις γραμμές που χρειάζονται αναστροφή
        src_is_A = (chunk["Src_IP"] == chunk["IP_A"])

        # Αν o Src είανι ο Α, τότε Port_A είναι το Src_Port, αλλιώς είναι το Dst_Port
        chunk["Port_A"] = np.where(src_is_A, chunk["Src_Port"], chunk["Dst_Port"])
        chunk["Port_B"] = np.where(src_is_A, chunk["Dst_Port"], chunk["Src_Port"])

        # Αν ο Src είναι ο Α:
        #   Bytes_A_to_B = Bytes_Out (γιατί φεύγουν από τον Src)
        #   Bytes_B_to_A = Bytes_In (γιατί έρχονται στον Src)
        # Και το αντίστροφο αν ο Src είναι ο B.
        chunk["Bytes_A_to_B"] = np.where(src_is_A, chunk["Bytes_Out"], chunk["Bytes_In"])
        chunk["Bytes_B_to_A"] = np.where(src_is_A, chunk["Bytes_In"], chunk["Bytes_Out"])

        # Ομαδοποίηση Chunk
        # ΣΗΜΕΙΩΣΗ: Ομαδοποιούμε και ανά Protocol. 
        # (Flows TCP A->B και UDP A->B θεωρούνται διαφορετικά flows).
        grouped = (
            chunk
            .groupby(["IP_A", "Port_A", "IP_B", "Port_B", "Protocol"], as_index=False)
            .agg(
                Bytes_A_to_B=("Bytes_A_to_B", "sum"),
                Bytes_B_to_A=("Bytes_B_to_A", "sum"),
                Flow_Count=("Protocol", "count")
            )
        )
        aggregated_chunks.append(grouped)

        # MEMORY OPTIMIZATION (FEEDBACK UPDATE):
        # Διαγράφουμε ρητά το raw chunk για να ελευθερώσουμε RAM πριν το επόμενο loop.
        chunk_len = len(chunk)

        del chunk
        del src_is_A
        print(f"  Επεξεργάστηκε chunk {i} με {chunk_len} γραμμές.")

    # Ενεργοποίηση Garbage Collector για απελευθέρωση μνήμης
    gc.collect()

    # Συνένωση όλων των ομαδοποιημένων chunks
    if not aggregated_chunks:
        print(f"  Κανένα δεδομένο για επεξεργασία στο αρχείο {file_name}.")
        continue

    print("Συνένωση όλων των ομαδοποιημένων chunks...")

    final = (
        pd.concat(aggregated_chunks, ignore_index=True)
        .groupby(["IP_A", "Port_A", "IP_B", "Port_B", "Protocol"], as_index=False)
        .agg(
            Bytes_A_to_B=("Bytes_A_to_B", "sum"),
            Bytes_B_to_A=("Bytes_B_to_A", "sum"),
            Flow_Count=("Flow_Count", "sum")
        )
    )

    # Διαγράφουμε τη λίστα για να ελευθερώσουμε μνήμη πριν την αποθήκευση
    del aggregated_chunks
    gc.collect()

    # Προσθήκη στήλης με το συνολικό αριθμό bytes
    final["Total_Bytes"] = final["Bytes_A_to_B"] + final["Bytes_B_to_A"]

    # --- FINAL SAFETY POLISH ---
    # Βεβαιωνόμαστε ότι τα Ports είναι int στο τελικό αρχείο (αντί για float)
    final["Port_A"] = final["Port_A"].astype(int)
    final["Port_B"] = final["Port_B"].astype(int)

    # Αποθήκευση του τελικού αρχείου
    output_path = os.path.join(OUTPUT_FOLDER, f"bidirectional_{file_name}")
    final.to_csv(output_path, index=False)

    print(f"Αποθηκεύτηκε το αρχείο: {output_path} με {len(final)} εγγραφές.")

#*---- ΤΕΛΟΣ ----
print("\nΟλοκληρώθηκε η επεξεργασία όλων των αρχείων.")
