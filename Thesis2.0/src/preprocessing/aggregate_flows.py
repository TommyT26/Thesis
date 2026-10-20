import os
import glob
import pandas as pd
import numpy as np

# Εισαγωγή ρυθμίσεων από το κεντρικό config
from src.config import PATHS

TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def normalize_protocol(series):
    """TCP / UDP / OTHER (δέχεται και αριθμητικά protocol numbers: 6 = TCP, 17 = UDP)."""
    p = series.astype(str).str.strip().str.upper()
    is_tcp = p.str.contains('TCP') | (p == '6')
    is_udp = p.str.contains('UDP') | (p == '17')
    return pd.Series(np.select([is_tcp, is_udp], ['TCP', 'UDP'], default='OTHER'), index=series.index)


def add_time_columns(df):
    """
    Σπάει τη στήλη Time σε Date / Hour / DayOfWeek και μετά την πετάει.

    Η Time ήταν η μόνη χρονική πληροφορία και την πέταγα εδώ, οπότε οι οκτώ
    ημέρες κατέληγαν σε ένα στατικό προφίλ ανά host. Κρατώ μικρούς ακέραιους
    αντί για datetime ή strings· σε αρχεία 4,5 εκατ. γραμμών η διαφορά στη
    μνήμη μετράει.
    """
    ts = pd.to_datetime(df.pop('Time'), format=TIME_FORMAT, errors='coerce')
    if ts.isna().all():
        raise ValueError("Καμία έγκυρη τιμή στη στήλη Time — έλεγξε τη μορφή των δεδομένων.")

    # Προσοχή: η ημερομηνία δεν χωράει σε float32. Το float32 έχει 24 bits
    # mantissa, δηλαδή κρατά ακριβώς ακεραίους μόνο μέχρι 16.777.216, οπότε
    # το 20251203 στρογγυλοποιείται και ημέρες συγχωνεύονται μεταξύ τους.
    # Οι nullable ακέραιοι της pandas κρατούν ακρίβεια και δέχονται NA.
    df['Date'] = (ts.dt.year * 10000 + ts.dt.month * 100 + ts.dt.day).astype('Int32')
    df['Hour'] = ts.dt.hour.astype('Int8')
    df['DayOfWeek'] = ts.dt.dayofweek.astype('Int8')  # 0 = Δευτέρα, 5-6 = Σαββατοκύριακο
    invalid = int(ts.isna().sum())
    del ts
    return df, invalid


def host_time_table(df):
    """
    Δραστηριότητα ανά host, ημέρα και ώρα.

    Κάθε ροή μετράει και για τα δύο άκρα της, όπως και στην εξαγωγή
    χαρακτηριστικών: ένας host είναι "ενεργός" είτε στέλνει είτε λαμβάνει.
    """
    cols = ['Date', 'Hour', 'DayOfWeek', 'Bytes']
    both = pd.concat([
        df[['Src_Host'] + cols].rename(columns={'Src_Host': 'Host'}),
        df[['Dst_Host'] + cols].rename(columns={'Dst_Host': 'Host'}),
    ], ignore_index=True)
    both = both.dropna(subset=['Host', 'Date', 'Hour'])

    return both.groupby(['Host', 'Date', 'Hour', 'DayOfWeek'], observed=True).agg(
        Flows=('Bytes', 'size'),
        Bytes=('Bytes', 'sum'),
    ).reset_index()


def run():
    print("\n--- Έναρξη: Συνένωση σε Bidirectional Flows ---")

    # Διαβάζει τα καθαρισμένα αρχεία από το 2_processed
    processed_files = glob.glob(os.path.join(PATHS['processed_data'], "clean_*.csv"))

    if not processed_files:
        print(f"Προειδοποίηση: Δεν βρέθηκαν αρχεία στον φάκελο {PATHS['processed_data']}")
        return

    for file_path in processed_files:
        # Μετονομάζουμε το clean_20251126.csv σε bidir_20251126.csv
        filename = os.path.basename(file_path).replace("clean_", "bidir_", 1)
        print(f"Επεξεργασία αρχείου: {filename}")

        # Διαβάζουμε μόνο τις στήλες που χρειαζόμαστε.
        # Src_Host/Dst_Host = οι hosts μετά την NAT επίλυση (βλ. clean_anonymize.py)
        df = pd.read_csv(
            file_path, low_memory=False,
            usecols=['Time', 'Protocol', 'Src_Host', 'Src_Port', 'Dst_Host', 'Dst_Port',
                     'Bytes']
        )

        # Καθαρισμός κενών χαρακτήρων από τα νούμερα και μετατροπή σε αριθμητικούς τύπους
        for col in ['Src_Port', 'Dst_Port', 'Bytes']:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce').fillna(0)

        df['Protocol'] = normalize_protocol(df['Protocol'])

        # --- ΧΡΟΝΙΚΗ ΔΙΑΣΤΑΣΗ ---
        df, invalid_times = add_time_columns(df)
        if invalid_times:
            print(f" -> Προειδοποίηση: {invalid_times} γραμμές με μη αναγνωρίσιμο χρόνο (αγνοούνται στα χρονικά).")

        print(" -> Υπολογισμός δραστηριότητας ανά host, ημέρα και ώρα...")
        temporal = host_time_table(df)
        temporal_path = os.path.join(PATHS['bidirectional_data'],
                                     filename.replace("bidir_", "temporal_", 1))
        temporal.to_csv(temporal_path, index=False)
        print(f" -> Αποθηκεύτηκε στο: {temporal_path} ({len(temporal)} γραμμές)")

        # --- BIDIRECTIONAL ΛΟΓΙΚΗ ---
        # Η κίνηση A->B και B->A ανήκει στο ίδιο flow. Ορίζουμε ως "A" το μικρότερο (αλφαβητικά) hash.
        # Η κατεύθυνση δεν χάνεται: κρατώ χωριστά bytes και flows για A->B και B->A.
        a_is_src = df['Src_Host'] < df['Dst_Host']
        df['IP_A'] = np.where(a_is_src, df['Src_Host'], df['Dst_Host'])
        df['IP_B'] = np.where(a_is_src, df['Dst_Host'], df['Src_Host'])

        # Θύρα υπηρεσίας: η μικρότερη από τις δύο. Η θύρα του client είναι συνήθως
        # τυχαία και αλλάζει σε κάθε σύνδεση, οπότε αν την κρατούσα στο groupby τα
        # flows δεν θα ενώνονταν ποτέ.
        #
        # Προσοχή στο Service_Port == 0: το ICMP και το IGMP δεν έχουν θύρες και
        # ο εξαγωγέας γράφει 0, οπότε το min() βγάζει 0. Είναι 39.870 ροές
        # (0,16% των 25,4 εκατ.) και 3.164 γραμμές εδώ. Το 0 ΔΕΝ είναι θύρα
        # υπηρεσίας· όπου μετράω «καταχωρημένες θύρες» πρέπει να εξαιρείται,
        # αλλιώς περνά ως <1024. Βλ. port_semantics.py.
        df['Service_Port'] = np.minimum(df['Src_Port'], df['Dst_Port']).astype('int64')

        df['Bytes_AtoB'] = np.where(a_is_src, df['Bytes'], 0)
        df['Bytes_BtoA'] = np.where(a_is_src, 0, df['Bytes'])
        df['Flows_AtoB'] = a_is_src.astype('int64')
        df['Flows_BtoA'] = (~a_is_src).astype('int64')

        print(" -> Ομαδοποίηση (Aggregation) ανά ζεύγος IP, θύρα υπηρεσίας και πρωτόκολλο...")
        grouped = df.groupby(['IP_A', 'IP_B', 'Service_Port', 'Protocol']).agg(
            Bytes_AtoB=('Bytes_AtoB', 'sum'),
            Bytes_BtoA=('Bytes_BtoA', 'sum'),
            Flows_AtoB=('Flows_AtoB', 'sum'),
            Flows_BtoA=('Flows_BtoA', 'sum'),
        ).reset_index()

        output_path = os.path.join(PATHS['bidirectional_data'], filename)
        grouped.to_csv(output_path, index=False)
        print(f" -> Αποθηκεύτηκε στο: {output_path}")

    print("--- Ολοκληρώθηκε ---")


if __name__ == "__main__":
    run()
