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
from src.models.peer_recurrence import behaviour_groups, MIN_PORTS

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


def unambiguous_service_ports(hosts):
    """
    Οι θύρες υπηρεσίας κάθε host, κρατώντας ΜΟΝΟ τις ροές όπου το min() δεν
    μπορεί να κάνει λάθος.

    ΓΙΑΤΙ ΧΡΕΙΑΖΕΤΑΙ. Το bidir κρατά μόνο το Service_Port, δηλαδή το
    αποτέλεσμα του min(), και όχι το ζεύγος από το οποίο προέκυψε. Έτσι το
    ποσοστό διαδοχικών υπολογίζεται πάνω σε αριθμούς που μπορεί να μην είναι
    καν θύρες υπηρεσίας. Εδώ ξαναδιαβάζονται τα clean αρχεία, που έχουν και
    τις δύο θύρες, και κρατιέται μόνο η περίπτωση «μία γνωστή θύρα»: ακριβώς
    μία από τις δύο είναι κάτω από το όριο και μη μηδενική. Τότε και μόνο τότε
    το min() είναι η θύρα υπηρεσίας.

    Αποκλείονται επίτηδες δύο περιπτώσεις. Όταν καμία θύρα δεν είναι
    καταχωρημένη, το min() είναι ο μικρότερος δύο εφήμερων αριθμών. Και όταν
    είναι καταχωρημένες ΚΑΙ ΟΙ ΔΥΟ, δεν υπάρχει τρόπος να ξεχωρίσει ποια είναι
    η υπηρεσία — και αυτή ακριβώς η περίπτωση παράγει το τεχνούργημα που
    περιγράφει η verify_candidates().
    """
    paths = sorted(glob.glob(os.path.join(PATHS['processed_data'], 'clean_*.csv')))
    wanted = set(hosts)
    ports = {host: set() for host in wanted}
    partners = {host: {} for host in wanted}

    for path in paths:
        for chunk in pd.read_csv(path, usecols=['Src_Port', 'Dst_Port',
                                                'Src_Host', 'Dst_Host'],
                                 chunksize=CHUNK_SIZE, low_memory=False):
            mine = chunk['Src_Host'].isin(wanted) | chunk['Dst_Host'].isin(wanted)
            if not mine.any():
                continue
            chunk = chunk[mine]
            src = pd.to_numeric(chunk['Src_Port'], errors='coerce').fillna(0)
            dst = pd.to_numeric(chunk['Dst_Port'], errors='coerce').fillna(0)
            low = np.minimum(src, dst).to_numpy()
            high = np.maximum(src, dst).to_numpy()

            known = ((low > 0) & (low < WELL_KNOWN_PORT_MAX)).astype(int) + \
                    ((high > 0) & (high < WELL_KNOWN_PORT_MAX)).astype(int)
            host = np.where(chunk['Src_Host'].isin(wanted),
                            chunk['Src_Host'], chunk['Dst_Host'])

            # Η αντίπερα θύρα κρατιέται για ΟΛΕΣ τις ροές, ακόμη και τις
            # διφορούμενες: αυτή απαντά «τι είναι στην πραγματικότητα αυτή η
            # κίνηση» όταν ένας υποψήφιος δεν επιβεβαιωθεί.
            for h, hi in zip(host, high):
                counter = partners[h]
                counter[int(hi)] = counter.get(int(hi), 0) + 1

            for h, p in zip(host[known == 1], low[known == 1]):
                ports[h].add(int(p))

    return ports, partners


def consecutive(values):
    """Ποσοστό διαδοχικών σε ένα σύνολο θυρών· NaN με λιγότερες από δύο."""
    arr = np.sort(np.fromiter(values, dtype=int)) if values else np.empty(0, int)
    return float((np.diff(arr) == 1).mean()) if len(arr) > 1 else float('nan')


def share_text(value):
    """Ποσοστό για εκτύπωση· «—» όταν δεν ορίζεται (καμία ροή)."""
    return '—' if value is None or value != value else f"{value:.1%}"


def verify_candidates(flagged, filename='sequential_verified.csv'):
    """
    Ξαναμετράει τη διαδοχικότητα των υποψηφίων χωρίς το τεχνούργημα του min().

    ΤΙ ΒΡΗΚΕ. Και οι 21 υποψήφιοι που σημαιοδοτούσε η προηγούμενη εκδοχή
    κατέρρευσαν: το σύνολο των θυρών υπηρεσίας τους πέφτει από 20-112 τιμές σε
    ΜΙΑ, την 123. Το 100% των ροών τους έχει τη θύρα 123 στη μία πλευρά,
    δηλαδή είναι αποκλειστικά NTP, και η συσκευή NAT ξαναγράφει τη θύρα
    προέλευσης σε διαδοχικές χαμηλές τιμές (Src_Port 123 -> NAT_Src_Port 1, 2,
    3 ... 122· μετρημένο στις 27/11, κάθε τιμή από 1 έως 122 εμφανίζεται). Το
    NAT γράφει και τιμές ΠΑΝΩ από το 123 (π.χ. 185), αλλά εκείνες δεν
    πειράζουν: το min() κρατά τότε το 123. Τεχνούργημα παράγουν μόνο οι
    τιμές κάτω από το 123, που το min() διαλέγει ως «θύρα υπηρεσίας».

    filename: πού γράφεται ο πίνακας. Κάθε καλών έχει το δικό του αρχείο —
    όταν ήταν κοινό, ο έλεγχος μίας ημέρας έσβηνε την επαλήθευση των 21.

    Τρία βήματα — κατανομή θυρών από το NAT, μετά min(), μετά ο δείκτης
    διαδοχικότητας — και βγαίνει σήμα σάρωσης από κίνηση χρονισμού. Το ότι το
    εύρος τους ήταν πάντα «1 έως 123» ήταν η υπογραφή του τεχνουργήματος: δεν
    μπορεί να ξεπεράσει το 123, γιατί το min() με το 123 δεν ξεπερνιέται.

    Το ίδιο τεχνούργημα σημαιοδοτεί και δύο εσωτερικούς hosts του 192.168.1.x
    με 131 και 175 «σαρωμένες» θύρες, αν αφαιρεθούν τα φίλτρα
    MIN_PORTS/MIN_DAYS του σταδίου peers.

    Ο έλεγχος μένει στον κώδικα ακόμη κι αν κάποτε βγει καθαρός: είναι ο λόγος
    που ξέρουμε ότι ο πίνακας δεν λέει ψέματα.
    """
    if flagged.empty:
        return pd.DataFrame()

    print(f"\n  ΕΠΑΛΗΘΕΥΣΗ ΤΩΝ {len(flagged)} ΥΠΟΨΗΦΙΩΝ")
    print("  Ξαναδιαβάζονται τα clean αρχεία για το ΖΕΥΓΟΣ θυρών κάθε ροής.")
    ports, partners = unambiguous_service_ports(flagged.index)

    rows = []
    for host in flagged.index:
        counter = partners.get(host, {})
        total = sum(counter.values())
        top, top_count = max(counter.items(), key=lambda kv: kv[1]) if counter else (0, 0)
        checked = ports.get(host, set())
        share = consecutive(checked)
        confirmed = bool(share == share and share >= SEQUENTIAL_ALERT)
        # Μηδέν ροές στα clean σημαίνει ότι ο host δεν βρέθηκε στις στήλες
        # Src_Host/Dst_Host, ΟΧΙ ότι δεν σαρώνει. Σε συνεπή εκτέλεση δεν
        # συμβαίνει (τα bidir βγαίνουν από τις ίδιες στήλες)· αν συμβεί, τα
        # clean άλλαξαν μετά τα bidir. Χωρίς αυτή τη διάκριση η απουσία
        # μετρούσε ως απόρριψη.
        if total == 0:
            status = 'Δεν ελέγχθηκε'
        elif confirmed:
            status = 'Επιβεβαιώνεται'
        else:
            status = 'Απορρίπτεται'
        rows.append({
            'Host': host,
            'Διαδοχικές_αρχικά': round(float(flagged.loc[host, 'Ποσοστό_Διαδοχικών']), 4),
            'Θύρες_αρχικά': int(flagged.loc[host, 'Πλήθος']),
            'Θύρες_χωρίς_αμφιβολία': len(checked),
            'Διαδοχικές_ελεγμένες': round(share, 4) if share == share else None,
            'Κυρίαρχη_αντίπερα_θύρα': int(top),
            'Μερίδιο_της': round(top_count / total, 4) if total else None,
            'Επιβεβαιώνεται': confirmed,
            'Κατάσταση': status,
        })
    table = pd.DataFrame(rows).set_index('Host')

    print(f"\n  {'host':<12}{'αρχικά':>9}{'θύρες':>8}{'ελεγμένες':>11}"
          f"{'θύρες':>8}{'αντίπερα':>10}{'μερίδιο':>9}")
    print('  ' + '-' * 67)
    for host, row in table.iterrows():
        checked = (f"{row['Διαδοχικές_ελεγμένες']:.3f}"
                   if row['Διαδοχικές_ελεγμένες'] is not None else '—')
        print(f"  {host[:10]:<12}{row['Διαδοχικές_αρχικά']:>9.3f}"
              f"{row['Θύρες_αρχικά']:>8}{checked:>11}"
              f"{row['Θύρες_χωρίς_αμφιβολία']:>8}"
              f"{row['Κυρίαρχη_αντίπερα_θύρα']:>10}{share_text(row['Μερίδιο_της']):>9}")

    confirmed = int(table['Επιβεβαιώνεται'].sum())
    unchecked = int((table['Κατάσταση'] == 'Δεν ελέγχθηκε').sum())
    print(f"\n  -> ΕΠΙΒΕΒΑΙΩΝΟΝΤΑΙ ΩΣ ΣΑΡΩΤΕΣ: {confirmed} από {len(table)}")
    if unchecked:
        print(f"  -> [!] ΔΕΝ ΕΛΕΓΧΘΗΚΑΝ: {unchecked} — δεν βρέθηκαν ροές τους στις "
              "στήλες Src_Host/Dst_Host των clean αρχείων.")
        print("      Σε συνεπή εκτέλεση αυτό δεν συμβαίνει: τα clean και τα bidir "
              "δεν είναι από την ίδια εκτέλεση. Ξανατρέξε από το στάδιο clean.")
    checked_rows = table[table['Κατάσταση'] != 'Δεν ελέγχθηκε']
    if confirmed == 0 and not checked_rows.empty:
        seen = checked_rows['Κυρίαρχη_αντίπερα_θύρα'].value_counts()
        print(f"     Κανένας. Σε {seen.iloc[0]} από τους {len(checked_rows)} η αντίπερα "
              f"θύρα είναι η {seen.index[0]} στη")
        print("     συντριπτική πλειονότητα των ροών: η «σάρωση» είναι το min() "
              "πάνω σε θύρες που")
        print("     έγραψε το NAT. Ο πίνακας των υποψηφίων είναι πίνακας "
              "τεχνουργημάτων, όχι σαρωτών.")

    path = os.path.join(output_dir(), filename)
    table.to_csv(path, encoding=CSV_ENCODING)
    print(f"\n  -> Η επαλήθευση αποθηκεύτηκε στο: {path}")
    return table


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
        print("\n     ΠΡΟΣΟΧΗ: ο πίνακας αυτός ΔΕΝ αρκεί για να πει κανείς «σάρωση».")
        print("     Το Ποσοστό_Διαδοχικών υπολογίζεται πάνω στο Service_Port, που είναι")
        print("     min(Src_Port, Dst_Port) — και το min() μπορεί να διαλέξει θύρα που")
        print("     έγραψε το NAT αντί για θύρα υπηρεσίας. Την απάντηση τη δίνει η")
        print("     verify_candidates() παρακάτω, που ξαναδιαβάζει τα ζεύγη θυρών.")

    flagged.to_csv(os.path.join(output_dir(), 'sequential_well_known.csv'),
                   encoding=CSV_ENCODING)
    verify_candidates(flagged)
    return flagged


def single_day_candidates():
    """
    Οι hosts που ο έλεγχος διαδοχικότητας δεν έφτανε ποτέ να δει.

    ΤΟ ΚΕΝΟ. Η hidden_scanners() διαβάζει τον πίνακα επανεμφάνισης, και εκεί
    μπαίνουν μόνο hosts με MIN_DAYS+ ενεργές ημέρες — λογικό, αφού η
    επανεμφάνιση συγκρίνει ημέρα με ημέρα και με μία ημέρα δεν ορίζεται. Η
    διαδοχικότητα όμως ΔΕΝ χρειάζεται δεύτερη ημέρα: μετριέται στο ενιαίο
    σύνολο θυρών του host. Οπότε όποιος εμφανίστηκε μία μόνο ημέρα έμενε
    ανέλεγκτος για κάτι που θα μπορούσε κάλλιστα να είχε ελεγχθεί.

    ΠΟΣΟΙ ΕΙΝΑΙ. Από τους 9.133 hosts με Well_Known_Ratio > 0,5, οι 8.995 έχουν
    λιγότερες από MIN_PORTS θύρες υπηρεσίας και δεν νοείται να σαρώνουν εύρος.
    Από τους 138 που απομένουν, οι 26 είναι εσωτερικοί και εξαιρούνται εξ
    ορισμού ως υποκείμενα, 78 περνούν από τη hidden_scanners(), και 30 έμεναν
    εδώ, στο κενό. Τριάντα σε 123.133 είναι λίγοι, αλλά η πρόταση «δεν υπάρχει
    επιβεβαιωμένη σάρωση» δεν αντέχει εξαίρεση που δεν έχει εξεταστεί.

    ΤΙ ΒΡΗΚΕ. Ένας ξεπερνά το κατώφλι, με 72,2% διαδοχικές σε 19 θύρες. Είναι
    όμως το ίδιο τεχνούργημα με τους 21: οι 526 από τις 552 ροές του πέφτουν
    στη θύρα 123 και οι θύρες του κινούνται στο 90-123, δηλαδή πελάτης NTP
    πίσω από NAT. Περνά κι αυτός από τη verify_candidates(), με τα ίδια
    κριτήρια και χωρίς καμία εξαίρεση — ένα αρνητικό εύρημα αξίζει όσο και η
    διαδικασία που το παρήγαγε.
    """
    from src.utils.anonymize import get_internal_hashes

    recurrence_path = os.path.join(BASE_DIR, 'results', 'peers',
                                   'recurrence_ports.csv')
    profiles_path = os.path.join(PATHS['features'], 'features_global.csv')
    bidir = sorted(glob.glob(os.path.join(PATHS['bidirectional_data'],
                                          'bidir_*.csv')))
    if not os.path.exists(recurrence_path) or not os.path.exists(profiles_path):
        print("\n  Δεν βρέθηκαν τα αποτελέσματα του σταδίου peers.")
        return None
    if not bidir:
        print("\n  Δεν βρέθηκαν αρχεία bidir_*.csv.")
        return None

    profiles = pd.read_csv(profiles_path)
    tested = set(pd.read_csv(recurrence_path, encoding=CSV_ENCODING)['Host'])
    internal = set(get_internal_hashes())

    eligible = profiles[(profiles['Well_Known_Ratio'] > 0.5)
                        & (profiles['Unique_Ports'] >= MIN_PORTS)]
    gap = eligible[~eligible['Host'].isin(tested)
                   & ~eligible['Host'].isin(internal)]
    print(f"\n  ΟΣΟΙ ΕΜΕΝΑΝ ΕΚΤΟΣ ΤΟΥ ΕΛΕΓΧΟΥ — {len(gap)} hosts")
    print(f"  (Well_Known_Ratio > 0,5, >= {MIN_PORTS} θύρες, εκτός πίνακα "
          f"επανεμφάνισης, μη εσωτερικοί)")
    if gap.empty:
        print("  -> κανένας· η κάλυψη ήταν ήδη πλήρης.")
        return pd.DataFrame()

    targets = set(gap['Host'])
    ports = {host: set() for host in targets}
    for path in bidir:
        frame = pd.read_csv(path, usecols=['IP_A', 'IP_B', 'Service_Port'])
        for column in ('IP_A', 'IP_B'):
            subset = frame[frame[column].isin(targets)]
            for host, group in subset.groupby(column)['Service_Port']:
                ports[host].update(int(p) for p in group)

    rows = []
    for host, found in ports.items():
        share = consecutive(found)
        rows.append({'Host': host, 'Πλήθος': len(found),
                     'Ποσοστό_Διαδοχικών': share})
    table = pd.DataFrame(rows).set_index('Host').sort_values(
        'Ποσοστό_Διαδοχικών', ascending=False)

    flagged = table[table['Ποσοστό_Διαδοχικών'] >= SEQUENTIAL_ALERT]
    print(f"  -> με ποσοστό διαδοχικών >= {SEQUENTIAL_ALERT:.0%}: "
          f"{len(flagged)} από {len(table)}")

    verify_candidates(flagged, filename='sequential_single_day.csv')
    return table


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
    single_day_candidates()

    print(f"\n -> Τα αποτελέσματα αποθηκεύτηκαν στο: {output_dir()}")
    print("--- Ολοκληρώθηκε ο Έλεγχος του Service_Port ---")


if __name__ == "__main__":
    run()
