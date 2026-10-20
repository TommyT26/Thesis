"""
Ανάλυση ευαισθησίας στο ποσοστό ανωμαλιών.

Το CONTAMINATION_RATE ήταν αυθαίρετο νούμερο του pipeline: το
είχα βάλει 1,5% χωρίς να μπορώ να το δικαιολογήσω, ακριβώς όπως πριν ήταν
γραμμένο σταθερά το k. Εδώ το αντιμετωπίζω με την ίδια λογική που
αντιμετώπισα και το k, αλλά το ερώτημα δεν είναι το ίδιο.

Για το k υπάρχουν κριτήρια που προτείνουν τιμή. Εδώ δεν υπάρχουν, και ο
λόγος είναι δομικός: το Isolation Forest βγάζει μια ΣΥΝΕΧΗ βαθμολογία ανά
host, και το contamination απλώς κόβει αυτή τη βαθμολογία σε ένα σημείο. Δεν
αλλάζει το μοντέλο, μόνο το πού μπαίνει το όριο. Άρα το ερώτημα που έχει
νόημα δεν είναι «ποιο ποσοστό είναι σωστό», αλλά:

  1. Υπάρχει φυσικό σημείο κοπής στην καμπύλη της βαθμολογίας; Το ψάχνω με
     τον ίδιο αλγόριθμο Kneedle που χρησιμοποιώ και για τον αγκώνα του k.
  2. Αλλάζουν τα συμπεράσματα αν η γραμμή μπει αλλού; Αυτό το μετράω με
     Adjusted Rand Index: ομαδοποιώ τους καθαρούς hosts σε κάθε ποσοστό και
     συγκρίνω με την ομαδοποίηση στο 1,5%. Αν το ARI μένει ψηλά, η επιλογή
     του ποσοστού δεν επηρεάζει το αποτέλεσμα και αυτό είναι ακριβώς το
     επιχείρημα που χρειάζομαι.

Το δάσος εκπαιδεύεται ΜΙΑ φορά και τα ποσοστά εφαρμόζονται ως ποσοστημόρια
της ίδιας βαθμολογίας. Αυτό είναι ισοδύναμο με επανεκπαίδευση σε κάθε
ποσοστό, επειδή το contamination στο sklearn επηρεάζει μόνο το κατώφλι και
όχι το ίδιο το δάσος — και το επαληθεύω στην αρχή αντί να το υποθέσω.
"""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score

from src.config import (PATHS, BASE_DIR, RANDOM_STATE, CONTAMINATION_RATE,
                        MACRO_CLUSTERS)
from src.utils.scaling import scale_features
from src.models.data_split import split_internal_external
from src.models.select_k import knee_point

# Το εύρος που δοκιμάζεται, εκατέρωθεν της τρέχουσας τιμής
RATES = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)

# Ελληνικές κεφαλίδες: χωρίς BOM το Excel στα Windows τις διαβάζει λάθος
CSV_ENCODING = 'utf-8-sig'


def output_dir():
    path = os.path.join(BASE_DIR, 'results', 'sensitivity')
    os.makedirs(path, exist_ok=True)
    return path


def anomaly_scores(X):
    """
    Βαθμολογία κανονικότητας ανά host, από ένα δάσος.

    Το score_samples δίνει αρνητικές τιμές όπου χαμηλότερο = πιο ανώμαλο.
    """
    forest = IsolationForest(contamination=CONTAMINATION_RATE,
                             random_state=RANDOM_STATE)
    forest.fit(X)
    return forest, forest.score_samples(X)


def verify_equivalence(forest, scores, X):
    """
    Επαληθεύει ότι το κόψιμο σε ποσοστημόριο ταυτίζεται με το fit_predict.

    Αν δεν ταυτιζόταν, όλη η ανάλυση θα σύγκρινε διαφορετικά μοντέλα αντί για
    διαφορετικά κατώφλια του ίδιου.
    """
    from_predict = forest.predict(X) == -1
    threshold = np.percentile(scores, CONTAMINATION_RATE * 100)
    from_threshold = scores <= threshold
    agreement = float((from_predict == from_threshold).mean())
    print(f" -> Έλεγχος ισοδυναμίας κατωφλιού και fit_predict: "
          f"{agreement:.4%} συμφωνία "
          f"({int(from_predict.sum()):,} έναντι {int(from_threshold.sum()):,} ανωμαλίες)")
    return agreement


def cluster(df, n_clusters):
    """K-Means στους καθαρούς hosts, όπως στη μακροσκοπική ανάλυση."""
    X = scale_features(df)
    labels = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE,
                    n_init=10).fit_predict(X)
    return pd.Series(labels, index=df['Host'].to_numpy())


def knee_of_scores(scores):
    """Το σημείο καμπής της ταξινομημένης καμπύλης βαθμολογίας, ως ποσοστό."""
    ordered = np.sort(scores)
    position = knee_point(np.arange(len(ordered)), ordered)
    return position / len(ordered)


def run():
    print("\n--- Έναρξη: Ευαισθησία στο Ποσοστό Ανωμαλιών ---")
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

    print(f" -> {len(external):,} εξωτερικοί hosts | τρέχουσα τιμή "
          f"{CONTAMINATION_RATE:.3%} | k={MACRO_CLUSTERS}")

    X = scale_features(external)
    forest, scores = anomaly_scores(X)
    verify_equivalence(forest, scores, X)

    knee = knee_of_scores(scores)
    print(f" -> Σημείο καμπής της καμπύλης βαθμολογίας (Kneedle): {knee:.3%}")

    reference = None
    rows = []
    for rate in RATES:
        threshold = np.percentile(scores, rate * 100)
        is_anomaly = scores <= threshold
        clean = external[~is_anomaly]
        anomalies = external[is_anomaly]
        labels = cluster(clean, MACRO_CLUSTERS)

        if np.isclose(rate, CONTAMINATION_RATE):
            reference = labels

        rows.append({
            'Ποσοστό': rate,
            'Ανωμαλίες': int(is_anomaly.sum()),
            'Καθαροί': int((~is_anomaly).sum()),
            'Κατώφλι_Σκορ': threshold,
            'Διάμ_Bytes_Ανωμαλιών': anomalies['Total_Bytes'].median(),
            'Διάμ_Flows_Ανωμαλιών': anomalies['Total_Flows'].median(),
            'Διάμ_Peers_Ανωμαλιών': anomalies['Unique_Peers'].median(),
            '_labels': labels,
        })

    if reference is None:
        print("Προειδοποίηση: Η τρέχουσα τιμή δεν είναι στο εύρος δοκιμής.")
        return

    for row in rows:
        labels = row.pop('_labels')
        common = labels.index.intersection(reference.index)
        row['Κοινοί_hosts'] = len(common)
        row['ARI_vs_Αναφοράς'] = adjusted_rand_score(
            labels.loc[common], reference.loc[common]) if len(common) > 1 else np.nan

    table = pd.DataFrame(rows)
    table[['Ανωμαλίες', 'Καθαροί', 'Κοινοί_hosts']] = \
        table[['Ανωμαλίες', 'Καθαροί', 'Κοινοί_hosts']].astype(int)

    print(f"\n{'ποσοστό':>9}{'ανωμαλίες':>12}{'κατώφλι':>11}{'ARI vs 1,5%':>13}"
          f"{'διάμ. bytes':>14}{'διάμ. ροές':>12}{'διάμ. peers':>13}")
    print('-' * 84)
    for _, row in table.iterrows():
        mark = '  <-- τρέχουσα' if np.isclose(row['Ποσοστό'], CONTAMINATION_RATE) else ''
        print(f"{row['Ποσοστό']:>8.2%}{row['Ανωμαλίες']:>12,.0f}{row['Κατώφλι_Σκορ']:>11.4f}"
              f"{row['ARI_vs_Αναφοράς']:>13.4f}{row['Διάμ_Bytes_Ανωμαλιών']:>14,.0f}"
              f"{row['Διάμ_Flows_Ανωμαλιών']:>12,.0f}{row['Διάμ_Peers_Ανωμαλιών']:>13,.0f}{mark}")

    path = os.path.join(output_dir(), 'contamination.csv')
    table.to_csv(path, index=False, encoding=CSV_ENCODING)
    print(f"\n -> Τα αποτελέσματα αποθηκεύτηκαν στο: {path}")
    print("--- Ολοκληρώθηκε η Ανάλυση Ευαισθησίας ---")


if __name__ == "__main__":
    run()
