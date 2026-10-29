"""
Επανεμφάνιση peers και θυρών: ο έλεγχος που ξεχωρίζει τη σάρωση από το P2P.

Το μοτίβο «πολλές θύρες, ελάχιστα bytes» το παράγουν και οι δύο συμπεριφορές,
οπότε τα χαρακτηριστικά θυρών από μόνα τους δεν αρκούν. Αυτό που τις χωρίζει
είναι αν ο host ξαναβρίσκει τα ΙΔΙΑ πράγματα από μέρα σε μέρα:

  - ο σαρωτής δειγματοληπτεί, άρα κάθε ημέρα συναντά καινούρια peers ή
    καινούριες θύρες και η επικάλυψη μεταξύ ημερών είναι χαμηλή,
  - ένας peer σε swarm ή ένας πελάτης υπηρεσίας ξαναμιλά στα ίδια.

Μετρώνται δύο όψεις, επειδή η σάρωση έχει δύο μορφές:

  επανεμφάνιση peers   οριζόντια σάρωση: πολλοί στόχοι, μία θύρα
  επανεμφάνιση θυρών   κατακόρυφη σάρωση: ένας στόχος, πολλές θύρες

Σε αυτό το δίκτυο η οριζόντια σάρωση ΔΕΝ μπορεί να φανεί: οι εσωτερικοί hosts
είναι 64, οπότε ένας εξωτερικός έχει διάμεσο έναν peer και η επανεμφάνιση
είναι τετριμμένα 1. Γι' αυτό κάθε μέτρο υπολογίζεται μόνο στους hosts όπου
ορίζεται με νόημα (MIN_PEERS / MIN_PORTS), και όχι σε ολόκληρο το σύνολο.

Για κάθε όψη υπολογίζονται:
  Λόγος_Επανάληψης  ποσοστό των peers/θυρών που εμφανίστηκαν σε >= 2 ημέρες
  Μέσος_Jaccard     μέση ομοιότητα Jaccard των ημερήσιων συνόλων
  Μέση_Επικάλυψη    |τομή| / |μικρότερο σύνολο|

Το Jaccard πέφτει όταν δύο ημέρες έχουν πολύ διαφορετικό πλήθος, ακόμη κι αν
όλα τα στοιχεία της μικρής ημέρας επανεμφανίζονται. Η επικάλυψη δεν το
τιμωρεί αυτό, οπότε απαντά πιο καθαρά στο «ξαναβρήκε τα ίδια;».

Η ομαδοποίηση των hosts γίνεται με κριτήρια ΑΝΕΞΑΡΤΗΤΑ από την επανεμφάνιση
(θύρες P2P, γνωστές θύρες), ώστε η σύγκριση να μην είναι κυκλική.
"""
import os
import glob
import itertools

import numpy as np
import pandas as pd

from src.config import PATHS, BASE_DIR
from src.utils.anonymize import get_internal_hashes

# Κάτω από δύο ημέρες δεν υπάρχει τίποτα να συγκριθεί
MIN_DAYS = 2

# Κάτω από αυτά τα όρια η επανεμφάνιση είναι τετριμμένη: με έναν peer ή μία
# θύρα, το να τον ξαναδείς δεν λέει τίποτα για τη συμπεριφορά.
MIN_PEERS = 5
MIN_PORTS = 5

# Ελληνικές κεφαλίδες: χωρίς BOM το Excel στα Windows τις διαβάζει λάθος
CSV_ENCODING = 'utf-8-sig'


def output_dir():
    path = os.path.join(BASE_DIR, 'results', 'peers')
    os.makedirs(path, exist_ok=True)
    return path


def day_of(bidir_path):
    return os.path.basename(bidir_path).replace("bidir_", "").replace(".csv", "")


def observations(bidir_files):
    """
    (Host, Peer, Θύρα, Ημέρα) για κάθε άκρο κάθε ροής, χωρίς διπλότυπα.

    Κάθε ροή δίνει δύο γραμμές, όπως και στην εξαγωγή χαρακτηριστικών: και τα
    δύο άκρα είναι hosts με τον απέναντι ως peer.
    """
    frames = []
    for path in bidir_files:
        flows = pd.read_csv(path, usecols=['IP_A', 'IP_B', 'Service_Port'])
        both = pd.concat([
            flows.rename(columns={'IP_A': 'Host', 'IP_B': 'Peer'}),
            flows.rename(columns={'IP_B': 'Host', 'IP_A': 'Peer'}),
        ], ignore_index=True).drop_duplicates()
        both['Ημέρα'] = day_of(path)
        frames.append(both)
        print(f"    - {os.path.basename(path)}: {len(both):,} παρατηρήσεις")
    return pd.concat(frames, ignore_index=True)


def recurrence(frame, column):
    """Πόσα διαφορετικά <column> ανά host, και πόσα σε >= 2 ημέρες."""
    per_value = (frame.groupby(['Host', column])['Ημέρα']
                 .nunique().reset_index(name='Ημέρες'))
    per_value['Επαναλ'] = per_value['Ημέρες'] >= 2
    grouped = per_value.groupby('Host')
    table = pd.DataFrame({
        'Πλήθος': grouped.size(),
        'Επαναλαμβανόμενα': grouped['Επαναλ'].sum(),
    })
    table['Λόγος_Επανάληψης'] = table['Επαναλαμβανόμενα'] / table['Πλήθος']
    return table


def set_similarities(frame, column, hosts):
    """Μέσο Jaccard και μέση επικάλυψη των ημερήσιων συνόλων, ανά host."""
    subset = frame[frame['Host'].isin(hosts)]
    by_day = subset.groupby(['Host', 'Ημέρα'])[column].apply(set)

    rows = []
    for host, sets_of_host in by_day.groupby(level=0):
        sets = list(sets_of_host)
        jaccard, overlap = [], []
        for first, second in itertools.combinations(sets, 2):
            intersection = len(first & second)
            jaccard.append(intersection / len(first | second))
            overlap.append(intersection / min(len(first), len(second)))
        if jaccard:
            rows.append({'Host': host,
                         'Ζεύγη_Ημερών': len(jaccard),
                         'Μέσος_Jaccard': sum(jaccard) / len(jaccard),
                         'Μέση_Επικάλυψη': sum(overlap) / len(overlap)})
    return pd.DataFrame(rows).set_index('Host')


def consecutive_share(frame, hosts):
    """
    Ποσοστό διαδοχικών θυρών στο σύνολο κάθε host.

    Η κατακόρυφη σάρωση διατρέχει εύρη (80, 81, 82, ...), ενώ το P2P και οι
    υπηρεσίες πέφτουν σε διάσπαρτες θύρες. Μετράται στο ενιαίο σύνολο όλων
    των ημερών, όχι ανά ημέρα.
    """
    subset = frame[frame['Host'].isin(hosts)]
    ports = subset.groupby('Host')['Service_Port'].apply(lambda s: np.sort(s.unique()))
    shares = ports.apply(
        lambda arr: float((np.diff(arr) == 1).mean()) if len(arr) > 1 else np.nan)
    return shares.rename('Ποσοστό_Διαδοχικών')


def behaviour_groups(profiles):
    """Ομάδες με κριτήρια ανεξάρτητα από την επανεμφάνιση."""
    p2p = profiles['P2P_Port_Ratio'] > 0
    well_known = profiles['Well_Known_Ratio'] > 0.5
    return [
        ('Με θύρες P2P', p2p),
        ('Κυρίως γνωστές θύρες (<1024)', ~p2p & well_known),
        ('Υψηλές τυχαίες θύρες', ~p2p & ~well_known),
    ]


def summarise(table, profiles, extra_columns=()):
    """Διάμεσες τιμές ανά ομάδα συμπεριφοράς."""
    merged = table.join(profiles, how='inner')
    rows = []
    for name, mask in behaviour_groups(merged):
        group = merged[mask]
        if group.empty:
            continue
        row = {
            'Ομάδα': name,
            'Hosts': len(group),
            'Διάμ_Ημέρες': group['Ημέρες'].median(),
            'Διάμ_Πλήθος': group['Πλήθος'].median(),
            'Διάμ_Λόγος_Επανάληψης': group['Λόγος_Επανάληψης'].median(),
            'Διάμ_Jaccard': group['Μέσος_Jaccard'].median(),
            'Διάμ_Επικάλυψη': group['Μέση_Επικάλυψη'].median(),
        }
        for column in extra_columns:
            row[f'Διάμ_{column}'] = group[column].median()
        rows.append(row)
    return pd.DataFrame(rows)


def report(title, note, summary):
    print(f"\n{'=' * 96}\n  {title}\n  {note}\n{'=' * 96}")
    if summary.empty:
        print("  Κανένας host δεν πληροί το κατώφλι.")
        return
    extra = [c for c in summary.columns if c.startswith('Διάμ_')
             and c not in ('Διάμ_Ημέρες', 'Διάμ_Πλήθος', 'Διάμ_Λόγος_Επανάληψης',
                           'Διάμ_Jaccard', 'Διάμ_Επικάλυψη')]
    header = (f"{'Ομάδα':<32}{'hosts':>8}{'ημέρες':>8}{'πλήθος':>8}"
              f"{'επανάλ.':>10}{'Jaccard':>10}{'επικάλ.':>10}")
    for column in extra:
        header += f"{column.replace('Διάμ_', ''):>14}"
    print(header)
    print('-' * len(header))
    for _, row in summary.iterrows():
        line = (f"{row['Ομάδα']:<32}{row['Hosts']:>8,}{row['Διάμ_Ημέρες']:>8.0f}"
                f"{row['Διάμ_Πλήθος']:>8,.0f}{row['Διάμ_Λόγος_Επανάληψης']:>10.3f}"
                f"{row['Διάμ_Jaccard']:>10.3f}{row['Διάμ_Επικάλυψη']:>10.3f}")
        for column in extra:
            line += f"{row[column]:>14.3f}"
        print(line)


def analyse(frame, column, min_count, days_per_host, profiles, extra=None):
    """Ένας πλήρης κύκλος: επανεμφάνιση, ομοιότητες, σύνοψη ανά ομάδα."""
    counts = recurrence(frame, column)
    eligible = counts[counts['Πλήθος'] >= min_count].index
    eligible = eligible.intersection(days_per_host[days_per_host >= MIN_DAYS].index)
    if len(eligible) == 0:
        return pd.DataFrame(), pd.DataFrame()

    table = counts.loc[eligible]
    table = table.join(set_similarities(frame, column, set(eligible)))
    table.insert(0, 'Ημέρες', days_per_host.loc[eligible])
    if extra is not None:
        table = table.join(extra)
    extra_columns = tuple(extra.name for extra in (extra,)) if extra is not None else ()
    return table, summarise(table, profiles, extra_columns)


def run():
    print("\n--- Έναρξη: Επανεμφάνιση Peers και Θυρών (σάρωση ή P2P;) ---")
    bidir_files = sorted(glob.glob(os.path.join(PATHS['bidirectional_data'], "bidir_*.csv")))
    if len(bidir_files) < MIN_DAYS:
        print(f"Προειδοποίηση: Χρειάζονται τουλάχιστον {MIN_DAYS} ημέρες. Παράλειψη.")
        return

    profiles_path = os.path.join(PATHS['features'], 'features_global.csv')
    if not os.path.exists(profiles_path):
        print("Προειδοποίηση: Δεν βρέθηκε το features_global.csv.")
        print("Τρέξε πρώτα:  python main.py --step features")
        return

    print(f" -> {len(bidir_files)} ημέρες")
    frame = observations(bidir_files)

    # Κρατάω μόνο τους εξωτερικούς hosts ως υποκείμενα της ανάλυσης.
    #
    # Το ερώτημα είναι αν hosts ΕΚΤΟΣ της σχολής σαρώνουν το δίκτυο ή κάνουν
    # P2P. Οι μηχανές της σχολής μόλυναν την ομάδα αναφοράς: στον πίνακα
    # θυρών ήταν 25 από τους 98 hosts των «γνωστών θυρών», δηλαδή το 26% της
    # ομάδας με την οποία συγκρίνονται οι ύποπτοι. Με την εξαίρεση, η ομάδα
    # πέφτει σε 73 hosts, η διάμεση επανάληψη από 0,544 σε 0,358 και η
    # επικάλυψη από 0,763 σε 0,521 (μετρημένο 21/10/2026).
    #
    # ΠΡΟΣΟΧΗ: η παραδοχή «οι εσωτερικοί δεν είναι υποψήφιοι σαρωτές» είναι
    # παραδοχή, όχι εύρημα. Δύο εσωτερικοί του 192.168.1.x περνούν το κατώφλι
    # διαδοχικότητας όταν αφαιρεθεί η εξαίρεση — αποδείχθηκαν κι αυτοί
    # τεχνούργημα NTP/NAT (verify_candidates στο port_semantics.py), αλλά η
    # εξαίρεση τούς έκρυβε ούτως ή άλλως. Μια ομάδα αναφοράς «κανονικής χρήσης υπηρεσιών»
    # που κατά ένα έκτο είναι η ίδια η υποδομή του εργαστηρίου δεν είναι
    # έγκυρο μέτρο σύγκρισης. Οι peers παραμένουν ως έχουν — εκεί ακριβώς
    # πρέπει να φαίνονται οι εσωτερικοί, ως στόχοι.
    internal = get_internal_hashes()
    before = frame['Host'].nunique()
    frame = frame[~frame['Host'].isin(internal)]
    removed = before - frame['Host'].nunique()

    days_per_host = frame.groupby('Host')['Ημέρα'].nunique()
    print(f" -> {removed} εσωτερικοί hosts εξαιρέθηκαν ως υποκείμενα "
          f"(παραμένουν ως peers)")
    print(f" -> {len(days_per_host):,} εξωτερικοί hosts, από τους οποίους "
          f"{(days_per_host >= MIN_DAYS).sum():,} σε {MIN_DAYS}+ ημέρες")

    profiles = pd.read_csv(profiles_path).set_index('Host')
    criteria = profiles[['P2P_Port_Ratio', 'Well_Known_Ratio']]

    peers_table, peers_summary = analyse(
        frame, 'Peer', MIN_PEERS, days_per_host, criteria)
    report("ΕΠΑΝΕΜΦΑΝΙΣΗ PEERS — οριζόντια σάρωση",
           f"μόνο hosts με >= {MIN_PEERS} peers σε {MIN_DAYS}+ ημέρες",
           peers_summary)

    ports_frame = frame[['Host', 'Service_Port', 'Ημέρα']].drop_duplicates()
    sequential = consecutive_share(
        ports_frame, set(days_per_host[days_per_host >= MIN_DAYS].index))
    ports_table, ports_summary = analyse(
        ports_frame, 'Service_Port', MIN_PORTS, days_per_host, criteria,
        extra=sequential)
    report("ΕΠΑΝΕΜΦΑΝΙΣΗ ΘΥΡΩΝ — κατακόρυφη σάρωση",
           f"μόνο hosts με >= {MIN_PORTS} θύρες σε {MIN_DAYS}+ ημέρες",
           ports_summary)

    for name, table, summary in (('peers', peers_table, peers_summary),
                                 ('ports', ports_table, ports_summary)):
        if not table.empty:
            table.to_csv(os.path.join(output_dir(), f'recurrence_{name}.csv'),
                         encoding=CSV_ENCODING)
            summary.to_csv(os.path.join(output_dir(), f'recurrence_{name}_groups.csv'),
                           index=False, encoding=CSV_ENCODING)

    print(f"\n -> Τα αποτελέσματα αποθηκεύτηκαν στο: {output_dir()}")
    print("--- Ολοκληρώθηκε η Ανάλυση Επανεμφάνισης ---")


if __name__ == "__main__":
    run()
