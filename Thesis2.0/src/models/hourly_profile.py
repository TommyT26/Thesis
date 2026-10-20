"""
Ωριαίο προφίλ δραστηριότητας ανά ομάδα.

Στη μικροσκοπική ανάλυση αποδίδω ρόλους στις ομάδες — σταθμοί εργασίας,
συσκευές συνεχούς λειτουργίας, servers, πύλη — και το επιχείρημα στηρίζεται σε
δύο αριθμούς ανά ομάδα, το Business_Ratio και τις Active_Hours. Ένας αριθμός
όμως είναι συνάθροιση: το 0,93 του Cluster 0 λέει πόση κίνηση έπεσε μέσα στο
ωράριο, δεν λέει ΠΩΣ κατανέμεται μέσα στο εικοσιτετράωρο. Δύο πολύ
διαφορετικές καμπύλες δίνουν το ίδιο Business_Ratio.

Εδώ βγάζω την καμπύλη. Ο πίνακας δραστηριότητας ανά host, ημέρα και ώρα
υπάρχει από τη Φάση 2 και δεν τον είχα αξιοποιήσει πουθενά — μόνο για να
βγάλω τα πέντε συναθροιστικά χαρακτηριστικά. Αν το Cluster 0 δείξει καμπάνα
στις εργάσιμες ώρες και το Cluster 1 ευθεία γραμμή, η απόδοση ρόλων παύει να
είναι εικασία από δύο δείκτες και γίνεται εικόνα.

Δύο μεθοδολογικές επιλογές που χρειάζονται δικαιολόγηση:

  1. Κανονικοποιώ ΑΝΑ HOST και μετά παίρνω μέσο όρο μέσα στην ομάδα, αντί να
     αθροίσω τις ροές όλης της ομάδας. Αλλιώς η καμπύλη της ομάδας είναι η
     καμπύλη του μεγαλύτερου host της: η πύλη έχει 18,3 εκατομμύρια ροές και
     θα κατάπινε οποιαδήποτε ομάδα βρισκόταν μαζί της. Έτσι κάθε host μετράει
     το ίδιο, που είναι και το νόημα του «προφίλ της ομάδας».

  2. Εξαιρώ τις μερικές ημέρες. Η 26/11 ξεκινά στις 08:36 και η ώρα 08 έχει
     μόλις 60 ροές από τα 2,3 εκατομμύρια της ημέρας, άρα ουσιαστικά της
     λείπουν οι ώρες πριν τις 09:00 — ακριβώς η πληροφορία που μετράω εδώ. Στα συναθροιστικά
     χαρακτηριστικά την κρατώ, επειδή η εξαίρεσή της θα άφηνε 8.376 hosts χωρίς
     χρονικά. Στην ωριαία καμπύλη το τίμημα είναι αντίστροφο: η μεροληψία
     πέφτει πάνω στο μετρούμενο μέγεθος. Η ασυμφωνία είναι σκόπιμη και τη
     δηλώνω. Ο εντοπισμός γίνεται αυτόματα (ημέρα με ελάχιστη ώρα > 0), όχι
     μόνο από το PARTIAL_DAYS του config.

Βγάζω και χωριστή καμπύλη μόνο για τις καθημερινές. Το Σαββατοκύριακο αλλάζει
σχήμα, και για την ταυτοποίηση ρόλων το σήμα είναι στις καθημερινές.
"""
import os
import glob

import numpy as np
import pandas as pd

from src.config import PATHS, BASE_DIR, PARTIAL_DAYS, BUSINESS_HOURS

# (επίπεδο, φάκελος, αρχείο ετικετών, τίτλος)
LEVELS = [
    ('macro', 'results_macro', 'clean_macro_global.csv', 'Εξωτερικοί hosts'),
    ('micro', 'results_micro', 'micro_global.csv', 'Εσωτερικοί hosts'),
]

HOURS = list(range(24))

# Ελληνικές κεφαλίδες: χωρίς BOM το Excel στα Windows τις διαβάζει λάθος
CSV_ENCODING = 'utf-8-sig'


def output_dir():
    path = os.path.join(BASE_DIR, 'results', 'hourly')
    os.makedirs(path, exist_ok=True)
    return path


def labels_of(folder_key, filename):
    """Host -> Cluster από τα αποτελέσματα της ομαδοποίησης."""
    path = os.path.join(PATHS[folder_key], filename)
    if not os.path.exists(path):
        return None
    table = pd.read_csv(path, usecols=['Host', 'Cluster'])
    return table.set_index('Host')['Cluster']


def partial_days(frames):
    """
    Ημέρες που δεν καλύπτουν όλο το 24ωρο.

    Τις βρίσκω από τα δεδομένα (ελάχιστη ώρα > 0) και όχι μόνο από το config,
    ώστε μια μελλοντική μερική ημέρα να μην περάσει απαρατήρητη.
    """
    found = set(int(d) for d in PARTIAL_DAYS)
    coverage = frames.groupby('Date')['Hour'].agg(['min', 'max'])
    for date, row in coverage.iterrows():
        if row['min'] > 0:
            found.add(int(date))
    return sorted(found), coverage


def load_temporal():
    """Ο πίνακας host x ημέρα x ώρα, από όλα τα temporal_*.csv."""
    paths = sorted(glob.glob(os.path.join(PATHS['bidirectional_data'], 'temporal_*.csv')))
    if not paths:
        return None
    frames = [pd.read_csv(p, usecols=['Host', 'Date', 'Hour', 'DayOfWeek', 'Flows'])
              for p in paths]
    return pd.concat(frames, ignore_index=True)


def hourly_shares(temporal, labels):
    """
    Μέσο ωριαίο προφίλ ανά ομάδα: πίνακας ομάδα x 24 ώρες, κάθε γραμμή αθροίζει 1.

    Κανονικοποίηση ανά host πρώτα (βλ. docstring του module), οπότε η τιμή
    διαβάζεται ως «τι ποσοστό της κίνησης ενός τυπικού host της ομάδας πέφτει
    σε αυτή την ώρα».
    """
    subset = temporal[temporal['Host'].isin(labels.index)]
    if subset.empty:
        return None, None

    per_host = (subset.groupby(['Host', 'Hour'])['Flows'].sum()
                .unstack(fill_value=0)
                .reindex(columns=HOURS, fill_value=0))

    totals = per_host.sum(axis=1)
    per_host = per_host[totals > 0]
    shares = per_host.div(totals[totals > 0], axis=0)
    if shares.empty:
        return None, None

    grouped = shares.groupby(labels.reindex(shares.index)).mean()
    grouped.index.name = 'Ομάδα'
    # Γυρίζω και το ποιοι hosts μπήκαν, ώστε τα πλήθη της σύνοψης να αφορούν
    # ΑΥΤΟΥΣ και όχι όλους όσους έχουν ετικέτα: με φίλτρο καθημερινών, ένας
    # host που εμφανίζεται μόνο Σαββατοκύριακο δεν συμμετέχει στην καμπύλη.
    return grouped, shares.index


def summarise(curve, hosts_by_cluster):
    """
    Τα νούμερα που διαβάζει κανείς από μια καμπύλη χωρίς να τη δει.

    Ο λόγος κορυφής προς ΔΙΑΜΕΣΟ και όχι προς πυθμένα: με πυθμένα, μία και
    μόνη νεκρή ώρα εκτοξεύει τον λόγο (το Cluster 0 των εσωτερικών έβγαζε 546
    αντί για 30) και, όταν ο πυθμένας είναι μηδέν, ο λόγος δεν ορίζεται
    καθόλου. Ο διάμεσος είναι ανθεκτικός και πάντα ορισμένος.

    Οι μηδενικές ώρες μπαίνουν ως χωριστή στήλη, επειδή λένε κάτι άλλο:
    διακοπτόμενη δραστηριότητα, όχι χαμηλή. Μια ομάδα με 15 νεκρές ώρες και
    κορυφή 20% δεν είναι «επίπεδη», είναι ριπές.
    """
    start, end = BUSINESS_HOURS
    business = curve[[h for h in HOURS if start <= h < end]].sum(axis=1)
    peak_hour = curve.idxmax(axis=1)
    rows = []
    for cluster in curve.index:
        values = curve.loc[cluster].to_numpy(dtype=float)
        median = float(np.median(values))
        rows.append({
            'Ομάδα': cluster,
            'Hosts': int(hosts_by_cluster.get(cluster, 0)),
            'Ώρα_κορυφής': int(peak_hour.loc[cluster]),
            'Εντός_ωραρίου': round(float(business.loc[cluster]), 4),
            'Κορυφή_προς_διάμεσο': (round(float(values.max() / median), 1)
                                     if median > 0 else np.nan),
            'Ώρες_χωρίς_κίνηση': int((values == 0).sum()),
        })
    return pd.DataFrame(rows)


def report(title, table, weekdays):
    scope = 'καθημερινές' if weekdays else 'όλες τις ημέρες'
    print(f"\n  {title} — {scope}")
    print(f"  {'ομάδα':>7}{'hosts':>8}{'κορυφή':>9}{'εντός ωραρίου':>16}"
          f"{'κορυφή/διάμεσο':>17}{'νεκρές ώρες':>14}")
    print('  ' + '-' * 69)
    for _, row in table.iterrows():
        ratio = ('              —' if pd.isna(row['Κορυφή_προς_διάμεσο'])
                 else f"{row['Κορυφή_προς_διάμεσο']:>17.1f}")
        print(f"  {int(row['Ομάδα']):>7}{int(row['Hosts']):>8,}"
              f"{int(row['Ώρα_κορυφής']):>7}:00"
              f"{row['Εντός_ωραρίου']:>16.1%}{ratio}"
              f"{int(row['Ώρες_χωρίς_κίνηση']):>14}")


def analyse(temporal, labels, level, title, weekdays):
    data = temporal[temporal['DayOfWeek'] <= 4] if weekdays else temporal
    curve, hosts = hourly_shares(data, labels)
    if curve is None:
        print(f"  Προειδοποίηση: Κανένας host του {level} στα χρονικά δεδομένα.")
        return

    counts = labels.reindex(hosts).value_counts()
    table = summarise(curve, counts)
    report(title, table, weekdays)

    suffix = 'weekdays' if weekdays else 'all'
    curve.round(6).to_csv(
        os.path.join(output_dir(), f'hourly_{level}_{suffix}.csv'),
        encoding=CSV_ENCODING)
    table.to_csv(
        os.path.join(output_dir(), f'hourly_{level}_{suffix}_summary.csv'),
        index=False, encoding=CSV_ENCODING)


def run():
    print("\n--- Έναρξη: Ωριαίο Προφίλ Δραστηριότητας ανά Ομάδα ---")

    temporal = load_temporal()
    if temporal is None:
        print("Προειδοποίηση: Δεν βρέθηκαν αρχεία temporal_*.csv.")
        print("Τρέξε πρώτα:  python main.py --step aggregate")
        return

    excluded, coverage = partial_days(temporal)
    print(f" -> {len(temporal):,} γραμμές host/ημέρα/ώρα σε {coverage.shape[0]} ημέρες")
    if excluded:
        print(f" -> Εξαιρούνται από την ωριαία καμπύλη οι μερικές ημέρες: "
              f"{', '.join(str(d) for d in excluded)}")
        print("    (λείπουν ώρες του 24ώρου, που είναι ακριβώς το μετρούμενο μέγεθος· "
              "στα συναθροιστικά χαρακτηριστικά παραμένουν)")
        temporal = temporal[~temporal['Date'].isin(excluded)]
        if temporal.empty:
            print("Προειδοποίηση: Δεν έμεινε καμία πλήρης ημέρα. Παράλειψη.")
            return

    seen = pd.Index(temporal['Host'].unique())
    for level, folder_key, filename, title in LEVELS:
        labels = labels_of(folder_key, filename)
        if labels is None:
            print(f"\n  Δεν βρέθηκε το {filename} (τρέξε το στάδιο {level}).")
            continue
        if labels.index.intersection(seen).empty:
            print(f"\n  Κανένας host του {level} δεν εμφανίζεται στα χρονικά "
                  f"δεδομένα. Παράλειψη.")
            continue
        for weekdays in (False, True):
            analyse(temporal, labels, level, title, weekdays)

    print(f"\n -> Τα αποτελέσματα αποθηκεύτηκαν στο: {output_dir()}")
    print("--- Ολοκληρώθηκε το Ωριαίο Προφίλ ---")


if __name__ == "__main__":
    run()
