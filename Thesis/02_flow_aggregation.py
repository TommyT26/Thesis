import pandas as pd
import glob
import os
import numpy as np

#*---- 1.ΡΥΘΜΙΣΕΙΣ ----

# Φάκελοι εισόδου και εξόδου
INPUT_FOLDER = 'processed_data/'
OUTPUT_FOLDER = 'bidirectional_data/' 

# Αριθμός γραμμών που θα διαβάζονται ανά chunk, προσαρμόστε ανάλογα με τη μνήμη σας
CHUNK_SIZE = 500000                     

# Δημιουργία φακέλου εξόδου αν δεν υπάρχει
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Oρισμός τύπων δεδομένων
DTYPES = {
    "Src_IP": "string",
    "Dst_IP": "string",
    "Protocol": "string",
    "Bytes_In": "floate64",
    "Bytes_Out": "float64",
}

# Εύρεση όλων των καθαρών αρχείων
files = glob.glob(os.path.join(INPUT_FOLDER, "clean_*.csv"))
print(f"Βρέθηκαν {len(files)} καθαρά αρχεία.")

#*---- 2.ΕΠΕΞΕΡΓΑΣΙΑ ΑΡΧΕΙΩΝ ----

for file_path in files:
    file_name = os.path.basename(file_path)
    print(f"\nΕπεξεργασία αρχείου: {file_name}")

    # Δημιουργία λίστας για την αποθήκευση των επεξεργασμένων chunks
    aggregated_chunks = []

    # low_memory=False για την αποφυγή προειδοποιήσεων τύπου
    reader = pd.read_csv(
        file_path,
        dtype=DTYPES,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        on_bad_lines='skip'
    )

    # Επεξεργασία κάθε chunk
    for i, chunk in enumerate(reader, start=1):

        # Καθαρισμός Ports και Bytes από τυχόν μη αριθμητικές τιμές
        chunk["Src_Port"] = pd.to_numeric(chunk["Src_Port"], errors='coerce').fillna(-1).astype(int)
        chunk["Dst_Port"] = pd.to_numeric(chunk["Dst_Port"], errors='coerce').fillna(-1).astype(int)

        chunk["Bytes_In"] = pd.to_numeric(chunk["Bytes_In"], errors='coerce').fillna(0)
        chunk["Bytes_Out"] = pd.to_numeric(chunk["Bytes_Out"], errors='coerce').fillna(0)

        # Πετάει τις γραμμές που δεν έχουν κίνηση
        chunk = chunk[(chunk["Bytes_In"] + chunk["Bytes_Out"]) > 0].copy()

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

        # Ομαδοποίηση βάσει των "κανονικοοποιημένων" IP/Ports
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
        print(f"  Επεξεργάστηκε chunk {i} με {len(chunk)} γραμμές.")

    # Συνένωση όλων των ομαδοποιημένων chunks
    if not aggregated_chunks:
        print(f"  Κανένα δεδομένο για επεξεργασία στο αρχείο {file_name}.")
        continue

    print("Συνένωση όλων των ομαδοποιημένων chunks...")

    final = (
        pd.contact(aggregated_chunks, ignore_index=True)
        .groupby(["IP_A", "Port_A", "IP_B", "Port_B", "Protocol"], as_index=False)
        .agg(
            Bytes_A_to_B=("Bytes_A_to_B", "sum"),
            Bytes_B_to_A=("Bytes_B_to_A", "sum"),
            Flow_Count=("Flow_Count", "sum")
        )
    )
    final["Total_Bytes"] = final["Bytes_A_to_B"] + final["Bytes_B_to_A"]

    # Αποθήκευση του τελικού αρχείου
    output_path = os.path.join(OUTPUT_FOLDER, f"bidirectional_{file_name}")
    final.to_csv(output_path, index=False)

    print(f"Αποθηκεύτηκε το αρχείο: {output_path} με {len(final)} εγγραφές.")

#*---- ΤΕΛΟΣ ----
print("\nΟλοκληρώθηκε η επεξεργασία όλων των αρχείων.")
