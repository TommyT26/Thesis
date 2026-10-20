"""
Συμφωνία Isolation Forest και DBSCAN στους ίδιους hosts.

Στη μακροσκοπική φάση ο DBSCAN τρέχει στο σύνολο ΜΕΤΑ την αφαίρεση των
ανωμαλιών, οπότε το noise του δεν είναι δεύτερη γνώμη για τους ύποπτους του
Isolation Forest: είναι hosts που το Isolation Forest έχει ήδη περάσει ως
κανονικούς. Για να συγκριθούν οι δύο μέθοδοι πρέπει να δουν το ίδιο σύνολο.

Εδώ τρέχουν και οι δύο σε ΟΛΟΥΣ τους εξωτερικούς hosts και βγαίνει πίνακας
συνάφειας 2x2. Δεν υπάρχει ground truth, άρα καμία από τις δύο δεν είναι η
"σωστή" απάντηση και δεν έχει νόημα να μιλήσω για recall. Αυτό που μετράω
είναι καθαρή συμφωνία:

  - Jaccard: η επικάλυψη των δύο συνόλων ως προς την ένωσή τους. Ο σωστός
    δείκτης εδώ, επειδή και τα δύο σύνολα είναι μικρά σε σχέση με το πλήθος
    των hosts και ένας δείκτης τύπου accuracy θα έβγαινε ~97% ακόμα και με
    μηδενική επικάλυψη, απλώς επειδή συμφωνούν στους κανονικούς.
  - Αναμενόμενη επικάλυψη υπό ανεξαρτησία: n * p_iso * p_dbscan. Αν η
    πραγματική επικάλυψη δεν ξεπερνά αυτήν, οι δύο μέθοδοι δείχνουν
    διαφορετικά πράγματα και δεν στηρίζει η μία την άλλη.

Προσοχή στην ερμηνεία: ο scaler προσαρμόζεται στο πλήρες εξωτερικό σύνολο,
όχι στο καθαρό, οπότε ο DBSCAN εδώ ΔΕΝ είναι ο ίδιος με αυτόν της
μακροσκοπικής φάσης (άλλη κλίμακα, άλλο eps, άλλο πλήθος ομάδων). Είναι
ξεχωριστή εκτέλεση με μόνο σκοπό τη σύγκριση των δύο κριτηρίων ανωμαλίας.
"""
import os

import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import PATHS, CONTAMINATION_RATE, RANDOM_STATE
from src.utils.scaling import scale_features
from src.models.data_split import split_internal_external
from src.models.dbscan_clustering import fit_dbscan

# Ελληνικές κεφαλίδες: χωρίς BOM το Excel στα Windows τις διαβάζει λάθος
CSV_ENCODING = 'utf-8-sig'


def flags(external):
    """
    (ανωμαλίες Isolation Forest, noise DBSCAN) ως boolean arrays, στο ίδιο σύνολο.

    Ένα scale_features για τα δύο μοντέλα, ώστε να μη διαφέρει η κλίμακα
    μεταξύ τους και η διαφορά να είναι καθαρά διαφορά κριτηρίου.
    """
    X = scale_features(external)

    iso = IsolationForest(contamination=CONTAMINATION_RATE,
                          random_state=RANDOM_STATE)
    iso_anomaly = iso.fit_predict(X) == -1

    db_labels, eps = fit_dbscan(X, name="macro_full")
    db_noise = db_labels == -1

    n_clusters = len(set(db_labels[db_labels != -1]))
    return iso_anomaly, db_noise, eps, n_clusters


def agreement(iso_anomaly, db_noise):
    """Πίνακας συνάφειας 2x2 και οι δείκτες συμφωνίας."""
    both = int((iso_anomaly & db_noise).sum())
    only_iso = int((iso_anomaly & ~db_noise).sum())
    only_db = int((~iso_anomaly & db_noise).sum())
    neither = int((~iso_anomaly & ~db_noise).sum())
    n = both + only_iso + only_db + neither

    union = both + only_iso + only_db
    jaccard = both / union if union else float('nan')

    # Πόση επικάλυψη θα περίμενα αν τα δύο κριτήρια ήταν ασύνδετα
    expected = n * (iso_anomaly.mean() * db_noise.mean())

    return {
        'n': n,
        'both': both,
        'only_iso': only_iso,
        'only_db': only_db,
        'neither': neither,
        'jaccard': jaccard,
        'share_of_iso': both / max(both + only_iso, 1),
        'share_of_db': both / max(both + only_db, 1),
        'expected_both': expected,
        'lift': both / expected if expected else float('nan'),
    }


def report(a, eps, n_clusters):
    print(f"\n -> DBSCAN στο πλήρες σύνολο: eps={eps:.3f} | "
          f"{n_clusters} ομάδες | noise {a['only_db'] + a['both']:,} hosts")
    print(f"\n{'':>26}{'DBSCAN noise':>15}{'DBSCAN ομάδα':>15}")
    print('-' * 56)
    print(f"{'IsoForest ανωμαλία':>26}{a['both']:>15,}{a['only_iso']:>15,}")
    print(f"{'IsoForest κανονικός':>26}{a['only_db']:>15,}{a['neither']:>15,}")
    print()
    print(f" -> Jaccard των δύο συνόλων: {a['jaccard']:.3f}")
    print(f" -> Από τις ανωμαλίες του IsoForest, ο DBSCAN θεωρεί noise το "
          f"{a['share_of_iso']:.1%}")
    print(f" -> Από το noise του DBSCAN, το IsoForest θεωρεί ανωμαλία το "
          f"{a['share_of_db']:.1%}")
    print(f" -> Αναμενόμενη επικάλυψη υπό ανεξαρτησία: {a['expected_both']:.0f} "
          f"hosts, δηλαδή x{a['lift']:.1f} περισσότερη από την τύχη")
    if a['lift'] < 2:
        print("    (τόσο χαμηλό lift σημαίνει ότι τα δύο κριτήρια δείχνουν "
              "διαφορετικά πράγματα, όχι ότι επιβεβαιώνουν το ένα το άλλο)")


def save(a, eps, n_clusters):
    table = pd.DataFrame({
        'Κατηγορία': ['IsoForest ανωμαλία (-1)', 'IsoForest κανονικός (+1)'],
        'DBSCAN_noise': [a['both'], a['only_db']],
        'DBSCAN_ομάδα': [a['only_iso'], a['neither']],
    })
    metrics = pd.DataFrame([{
        'Εξωτερικοί_hosts': a['n'],
        'Ανωμαλίες_IsoForest': a['both'] + a['only_iso'],
        'Noise_DBSCAN': a['both'] + a['only_db'],
        'Κοινοί': a['both'],
        'Jaccard': round(a['jaccard'], 4),
        'Ποσοστό_IsoForest_ως_noise': round(a['share_of_iso'], 4),
        'Ποσοστό_noise_ως_ανωμαλία': round(a['share_of_db'], 4),
        'Αναμενόμενη_επικάλυψη': round(a['expected_both'], 1),
        'Lift': round(a['lift'], 2),
        'DBSCAN_eps': round(float(eps), 4) if eps else None,
        'DBSCAN_ομάδες': n_clusters,
    }])

    path = os.path.join(PATHS['results_macro'], 'contingency_iso_dbscan.csv')
    table.to_csv(path, index=False, encoding=CSV_ENCODING)
    metrics.to_csv(path.replace('.csv', '_metrics.csv'), index=False,
                   encoding=CSV_ENCODING)
    print(f"\n -> Αποθηκεύτηκαν στο: {path}")
    print(f" -> και στο: {path.replace('.csv', '_metrics.csv')}")


def run():
    print("\n--- Έναρξη: Συμφωνία Isolation Forest και DBSCAN ---")
    profiles_path = os.path.join(PATHS['features'], 'features_global.csv')
    if not os.path.exists(profiles_path):
        print("Προειδοποίηση: Δεν βρέθηκε το features_global.csv.")
        print("Τρέξε πρώτα:  python main.py --step features")
        return

    profiles = pd.read_csv(profiles_path)
    _, external = split_internal_external(profiles)
    if len(external) < 100:
        print("Προειδοποίηση: Πολύ λίγοι εξωτερικοί hosts. Παράλειψη.")
        return

    print(f" -> {len(external):,} εξωτερικοί hosts | contamination "
          f"{CONTAMINATION_RATE:.2%}")

    iso_anomaly, db_noise, eps, n_clusters = flags(external)
    if eps is None:
        print("Προειδοποίηση: Ο DBSCAN δεν ολοκληρώθηκε. Η σύγκριση παραλείπεται.")
        return

    a = agreement(iso_anomaly, db_noise)
    report(a, eps, n_clusters)
    save(a, eps, n_clusters)
    print("\n--- Ολοκληρώθηκε η Σύγκριση Μεθόδων ---")


if __name__ == "__main__":
    run()
