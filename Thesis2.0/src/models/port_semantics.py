"""
Έλεγχος του Service_Port: τι πραγματικά μετράει η μικρότερη από τις δύο θύρες.

Στην ανάλυση επανεμφάνισης ορίζω Service_Port = min(Src_Port, Dst_Port) και
πάνω σε αυτό χτίζεται ο διαχωρισμός σε «γνωστές θύρες» και «υψηλές τυχαίες».
Στην αναφορά έχω γράψει την επιφύλαξη: αν κάποιος σάρωνε χαμηλές
θύρες, το min() θα τον έστελνε στην ομάδα των γνωστών θυρών, δηλαδή ακριβώς
στην ομάδα αναφοράς με την οποία συγκρίνω τους ύποπτους. Αυτό το module
μετατρέπει την επιφύλαξη σε μέτρηση.

Δύο ερωτήματα:

  1. Πόσο συχνά το min() είναι όντως θύρα υπηρεσίας; Σε μια κανονική σύνδεση
     πελάτη-διακομιστή η μία θύρα είναι καταχωρημένη και η άλλη εφήμερη, άρα
     το min() διαλέγει σωστά. Όταν και οι δύο θύρες είναι υψηλές — η τυπική
     περίπτωση P2P — το min() δεν είναι θύρα υπηρεσίας, είναι απλώς ο
     μικρότερος από δύο τυχαίους αριθμούς. Το μετράω σε όλες τις ροές.

  2. Κρύβεται σαρωτής μέσα στην ομάδα των γνωστών θυρών; Μια κατακόρυφη
     σάρωση διατρέχει εύρος θυρών, άρα αφήνει υψηλό ποσοστό διαδοχικών
     αριθμών. Κοιτάω την κατανομή του ποσοστού διαδοχικών μέσα στην ομάδα
     αναφοράς και, για όσους ξεχωρίζουν, βλέπω αν συμπεριφέρονται σαν
     σαρωτές ή σαν σταθεροί πελάτες υπηρεσιών.

Το πρώτο μέρος διαβάζει τα 4,3 GB των καθαρισμένων αρχείων σε κομμάτια: δεν
χρειάζεται να χωρέσουν στη μνήμη, μόνο να μετρηθούν.
"""
import os
import glob

import numpy as np
import pandas as pd

from src.config import (PATHS, BASE_DIR, CHUNK_SIZE, WELL_KNOWN_PORT_MAX)
from src.models.peer_recurrence import behaviour_groups

# Πάνω από αυτό το ποσοστό διαδοχικών θυρών, ο host αξίζει να κοιταχτεί
# χωριστά: τυχαία επιλογή θυρών δεν παράγει διαδοχικούς αριθμούς.
SEQUENTIAL_ALERT = 0.5

# Το ICMP και το IGMP δεν έχουν θύρες και ο εξαγωγέας γράφει 0. Το 0 δεν είναι
# θύρα υπηρεσίας, οπότε αυτές οι ροές μετριούνται χωριστά.
NO_PORTS = 'Χωρίς θύρες (ICMP/IGMP)'

# Ελληνικές κεφαλίδες: χωρίς BOM το Excel στα Windows τις διαβάζει λάθος
CSV_ENCODING = 'utf-8-sig'


def output_dir():
    path = os.path.join(BASE_DIR, 'results', 'ports')
    os.makedirs(path, exist_ok=True)
    return path


def port_pair_cases():
    """
    Πόσες ροές έχουν καμία, μία ή δύο καταχωρημένες θύρες.

    Γυρίζει (πίνακας μετρήσεων, σύνολο ροών, ροές με θύρες). Η ανάγνωση γίνεται
    σε κομμάτια και κρατιούνται μόνο οι δύο στήλες των θυρών.
    """
    paths = sorted(glob.glob(os.path.join(PATHS['processed_data'], 'clean_*.csv')))
    if not paths:
        return None, 0, 0

    counts = {NO_PORTS: 0,
              'Καμία γνωστή θύρα': 0, 'Μία γνωστή θύρα': 0, 'Δύο γνωστές θύρες': 0}
    total = 0
    for path in paths:
        rows = 0
        for chunk in pd.read_csv(path, usecols=['Src_Port', 'Dst_Port'],
                                 chunksize=CHUNK_SIZE, low_memory=False):
            src = pd.to_numeric(chunk['Src_Port'], errors='coerce').fillna(0)
            dst = pd.to_numeric(chunk['Dst_Port'], errors='coerce').fillna(0)

            # Χωρίς αυτή τη γραμμή το 0 μετριόταν ως «καταχωρημένη θύρα», αφού
            # 0 < 1024: 39.870 ροές ICMP/IGMP (0,16%) έμπαιναν στη λάθος στήλη.
            # Η επίδραση στα ποσοστά είναι μικρή, αλλά ο ορισμός πρέπει να
            # στέκει: το 0 σημαίνει «χωρίς θύρα», όχι «θύρα υπηρεσίας».
            portless = (src == 0) | (dst == 0)
            counts[NO_PORTS] += int(portless.sum())
            src, dst = src[~portless], dst[~portless]

            low = (src < WELL_KNOWN_PORT_MAX).astype(int) + \
                  (dst < WELL_KNOWN_PORT_MAX).astype(int)
            counts['Καμία γνωστή θύρα'] += int((low == 0).sum())
            counts['Μία γνωστή θύρα'] += int((low == 1).sum())
            counts['Δύο γνωστές θύρες'] += int((low == 2).sum())
            rows += len(chunk)
        total += rows
        print(f"    - {os.path.basename(path)}: {rows:,} ροές")

    # Τα ποσοστά των τριών περιπτώσεων υπολογίζονται πάνω στις ροές ΜΕ θύρες:
    # το ερώτημα «τι διαλέγει το min()» δεν ορίζεται όταν δεν υπάρχουν θύρες.
    with_ports = total - counts[NO_PORTS]
    rows = []
    for name, value in counts.items():
        base = total if name == NO_PORTS else with_ports
        rows.append({'Περίπτωση': name, 'Ροές': value,
                     'Ποσοστό': round(value / base, 4) if base else np.nan})
    return pd.DataFrame(rows), total, with_ports


def report_pairs(table, total, with_ports):
    print(f"\n  ΤΙ ΔΙΑΛΕΓΕΙ ΤΟ min(Src_Port, Dst_Port) — {total:,} ροές")
    print(f"  {'περίπτωση':<28}{'ροές':>16}{'ποσοστό':>11}")
    print('  ' + '-' * 55)
    for _, row in table.iterrows():
        print(f"  {row['Περίπτωση']:<28}{row['Ροές']:>16,}{row['Ποσοστό']:>11.2%}")
    print(f"\n     (τα τρία τελευταία ποσοστά είναι επί των {with_ports:,} ροών "
          f"που έχουν θύρες)")

    def share(name):
        return table.loc[table['Περίπτωση'] == name, 'Ποσοστό'].iloc[0]

    print(f"\n  -> Στο {share('Μία γνωστή θύρα'):.1%} των ροών το min() διαλέγει "
          f"τη μοναδική καταχωρημένη θύρα, δηλαδή σωστά.")
    print(f"  -> Στο {share('Καμία γνωστή θύρα'):.1%} καμία θύρα δεν είναι "
          f"καταχωρημένη: εκεί το Service_Port δεν είναι")
    print(f"     θύρα υπηρεσίας, είναι ο μικρότερος δύο εφήμερων αριθμών — και "
          f"αυτές ακριβώς οι ροές")
    print(f"     φτιάχνουν την ομάδα «υψηλές τυχαίες θύρες», οπότε η ετικέτα "
          f"της περιγράφει σωστά το τι είναι.")


def hidden_scanners():
    """
    Ψάχνει σαρωτές μέσα στην ομάδα αναφοράς των γνωστών θυρών.

    Χρησιμοποιεί τα ήδη υπολογισμένα αποτελέσματα του σταδίου peers, με τα
    ΙΔΙΑ κριτήρια ομαδοποίησης (εισαγωγή της behaviour_groups, ώστε να μην
    ξαναγραφτεί ο ορισμός και αποκλίνουν).

    Το κριτήριο που είχα γράψει αρχικά εδώ ήταν λάθος: έλεγε ότι υψηλές
    διαδοχικές ΚΑΙ υψηλή επανάληψη σημαίνει υπηρεσία και όχι σάρωση. Το
    διέψευσαν τα ίδια τα δεδομένα — οι δύο κορυφαίοι αυτού του πίνακα έχουν
    επανάληψη 0,99 και είναι σαρωτές, απλώς επίμονοι: ξαναδιατρέχουν το ίδιο
    πλήρες εύρος κάθε ημέρα. Η επανάληψη ξεχωρίζει το «σαρώνει άλλα κάθε
    φορά» από το «σαρώνει τα ίδια κάθε φορά», όχι τη σάρωση από την υπηρεσία.
    Αυτό το κρίνουν μόνο οι τιμές των θυρών, και γι' αυτό το module τυπώνει
    τους ύποπτους αντί να βγάζει μόνος του ετικέτα.
    """
    recurrence_path = os.path.join(BASE_DIR, 'results', 'peers',
                                   'recurrence_ports.csv')
    profiles_path = os.path.join(PATHS['features'], 'features_global.csv')
    if not os.path.exists(recurrence_path) or not os.path.exists(profiles_path):
        print("\n  Δεν βρέθηκαν τα αποτελέσματα του σταδίου peers.")
        print("  Τρέξε πρώτα:  python main.py --step peers")
        return None

    table = pd.read_csv(recurrence_path, encoding=CSV_ENCODING).set_index('Host')
    profiles = pd.read_csv(profiles_path).set_index('Host')
    criteria = profiles.reindex(table.index)

    masks = dict(behaviour_groups(criteria))
    well_known = table[masks['Κυρίως γνωστές θύρες (<1024)'].to_numpy()]
    if well_known.empty:
        print("\n  Η ομάδα των γνωστών θυρών είναι άδεια.")
        return None

    share = well_known['Ποσοστό_Διαδοχικών']
    print(f"\n  ΔΙΑΔΟΧΙΚΕΣ ΘΥΡΕΣ ΜΕΣΑ ΣΤΗΝ ΟΜΑΔΑ ΑΝΑΦΟΡΑΣ — {len(well_known)} hosts")
    print(f"  {'ποσοστημόριο':<16}{'ποσοστό διαδοχικών':>22}")
    print('  ' + '-' * 38)
    for q in (0.50, 0.75, 0.90, 0.95, 1.00):
        print(f"  {q:>13.0%}  {share.quantile(q):>22.3f}")

    flagged = well_known[share >= SEQUENTIAL_ALERT]
    print(f"\n  -> hosts με ποσοστό διαδοχικών >= {SEQUENTIAL_ALERT:.0%}: "
          f"{len(flagged)} από {len(well_known)}")
    if not flagged.empty:
        print(f"\n  {'θύρες':>8}{'ημέρες':>9}{'διαδοχικές':>13}"
              f"{'επανάληψη':>12}{'επικάλυψη':>12}")
        print('  ' + '-' * 54)
        for _, row in flagged.sort_values('Ποσοστό_Διαδοχικών',
                                          ascending=False).head(10).iterrows():
            print(f"  {int(row['Πλήθος']):>8}{int(row['Ημέρες']):>9}"
                  f"{row['Ποσοστό_Διαδοχικών']:>13.3f}"
                  f"{row['Λόγος_Επανάληψης']:>12.3f}"
                  f"{row['Μέση_Επικάλυψη']:>12.3f}")
        print("\n     Πώς διαβάζεται ο πίνακας. Υψηλές διαδοχικές ΚΑΙ χαμηλή "
              "επανάληψη σημαίνει ότι ο host")
        print("     σαρώνει άλλο υποσύνολο κάθε ημέρα — το κλασικό αποτύπωμα "
              "σάρωσης. Υψηλές διαδοχικές")
        print("     ΚΑΙ υψηλή επανάληψη είναι διφορούμενο και ΔΕΝ αρκεί για να "
              "αποκλειστεί η σάρωση:")
        print("     ταιριάζει και σε σταθερό σύνολο γειτονικών θυρών υπηρεσίας, "
              "και σε επίμονο σαρωτή")
        print("     που ξαναδιατρέχει το ίδιο πλήρες εύρος κάθε μέρα. Το κρίνουν "
              "οι ίδιες οι τιμές των")
        print("     θυρών: σε αυτά τα δεδομένα οι δύο κορυφαίοι (επανάληψη 0,99) "
              "χτυπούν ~117 διαδοχικές")
        print("     θύρες UDP μέσα στο εύρος 1–123, που δεν αντιστοιχούν σε "
              "καμία εφαρμογή — σάρωση.")

    flagged.to_csv(os.path.join(output_dir(), 'sequential_well_known.csv'),
                   encoding=CSV_ENCODING)
    return flagged


def run():
    print("\n--- Έναρξη: Έλεγχος Σημασίας του Service_Port ---")

    table, total, with_ports = port_pair_cases()
    if table is None:
        print("Προειδοποίηση: Δεν βρέθηκαν αρχεία clean_*.csv.")
        print("Τρέξε πρώτα:  python main.py --step clean")
        return
    report_pairs(table, total, with_ports)
    table.to_csv(os.path.join(output_dir(), 'port_pair_cases.csv'),
                 index=False, encoding=CSV_ENCODING)

    hidden_scanners()

    print(f"\n -> Τα αποτελέσματα αποθηκεύτηκαν στο: {output_dir()}")
    print("--- Ολοκληρώθηκε ο Έλεγχος του Service_Port ---")


if __name__ == "__main__":
    run()
