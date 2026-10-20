"""
Ευστάθεια των ομάδων ανά ημέρα.

Θέλω να απαντήσω αν οι ομάδες που βρίσκει ο K-Means περιγράφουν πραγματική
δομή, ή αν είναι απλώς μια διαμέριση που επιβάλλει ο αλγόριθμος.

Ομαδοποιώ κάθε ημέρα χωριστά, με τα δικά της δεδομένα και τον ίδιο αριθμό
ομάδων. Αν η δομή είναι πραγματική, ένας host που την Τρίτη βρέθηκε μαζί με
κάποιους άλλους θα βρεθεί μαζί τους και την Τετάρτη. Αυτό το μετράει το
Adjusted Rand Index στους κοινούς hosts δύο ημερών: 0 σημαίνει συμφωνία στο
επίπεδο της τύχης, 1 ταυτόσημη ομαδοποίηση.

Είναι ανεξάρτητο επιχείρημα από τους δείκτες εγκυρότητας. Εκείνοι μετρούν τη
γεωμετρία μιας ομαδοποίησης, αυτό την αναπαραγωγιμότητά της σε νέα δεδομένα.

Εξαιρώ δύο χαρακτηριστικά που εκφυλίζονται μέσα σε μία ημέρα: το Active_Days
(πάντα 1) και το Weekend_Ratio (πάντα 0 ή 1).
"""
import os
import glob
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from src.config import (PATHS, BASE_DIR, RANDOM_STATE,
                        MACRO_CLUSTERS, MICRO_CLUSTERS,
                        LOG_FEATURES, LINEAR_FEATURES, RATIO_FEATURES)
from src.utils.scaling import scale_features
from src.features.extract_features import build_profiles
from src.models.data_split import split_internal_external

# Χαρακτηριστικά χωρίς νόημα μέσα σε μία μόνο ημέρα
DEGENERATE_DAILY = {'Active_Days', 'Weekend_Ratio'}

# Οι κεφαλίδες εδώ είναι ελληνικές. Χωρίς BOM το Excel στα Windows τις διαβάζει
# ως Windows-1252 και βγάζουν αλαμπουρνέζικα, οπότε γράφω utf-8-sig.
CSV_ENCODING = 'utf-8-sig'

# Τα αποτελέσματα της συνολικής ομαδοποίησης, που χρησιμεύουν ως σημείο
# αναφοράς για την αντιστοίχιση των ετικετών: έτσι το «Cluster 3» μιας ημέρας
# σημαίνει το ίδιο πράγμα με το «Cluster 3» της αναφοράς και των πινάκων.
REFERENCE_FILES = {
    'macro': ('macro_clustering', 'clean_macro_global.csv'),
    'micro': ('micro_clustering', 'micro_global.csv'),
}

DAILY_LOG = [c for c in LOG_FEATURES if c not in DEGENERATE_DAILY]
DAILY_LINEAR = [c for c in LINEAR_FEATURES if c not in DEGENERATE_DAILY]
DAILY_RATIO = [c for c in RATIO_FEATURES if c not in DEGENERATE_DAILY]


def output_dir():
    path = os.path.join(BASE_DIR, 'results', 'stability')
    os.makedirs(path, exist_ok=True)
    return path


def day_of(bidir_path):
    return os.path.basename(bidir_path).replace("bidir_", "").replace(".csv", "")


def cluster_day(profiles, n_clusters):
    """
    Ομαδοποίηση των προφίλ μιας ημέρας. Επιστρέφει Series host -> ετικέτα.

    ΠΡΟΣΟΧΗ: εδώ ΔΕΝ αφαιρούνται οι ανωμαλίες, σε αντίθεση με τη μακροσκοπική
    φάση που τρέχει Isolation Forest πρώτα. Κάθε ημέρα ομαδοποιείται με τις
    δικές της ακραίες τιμές μέσα. Αυτό μετράει σε βάρος του αποτελέσματος, όχι
    υπέρ του: οι ακραίες τιμές τραβούν τα κέντρα του K-Means και ρίχνουν τη
    συμφωνία μεταξύ ημερών, άρα το ARI που βγαίνει είναι συντηρητική εκτίμηση
    της ευστάθειας.
    """
    if len(profiles) < n_clusters:
        return None
    X = scale_features(profiles, log_columns=DAILY_LOG,
                       linear_columns=DAILY_LINEAR, ratio_columns=DAILY_RATIO)
    labels = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE,
                    n_init=10).fit_predict(X)
    return pd.Series(labels, index=profiles.index)


def pairwise_ari(labels_by_day):
    """ARI για κάθε ζεύγος ημερών, στους hosts που εμφανίζονται και στις δύο."""
    rows = []
    for day_a, day_b in itertools.combinations(sorted(labels_by_day), 2):
        a, b = labels_by_day[day_a], labels_by_day[day_b]
        common = a.index.intersection(b.index)
        if len(common) < 2:
            continue
        rows.append({'Ημέρα_Α': day_a, 'Ημέρα_Β': day_b, 'Κοινοί_hosts': len(common),
                     'ARI': adjusted_rand_score(a.loc[common], b.loc[common])})
    return pd.DataFrame(rows)


def consistency(labels_by_day):
    """
    Για κάθε host: σε πόσες ημέρες εμφανίζεται και τι ποσοστό αυτών έπεσε στην
    ίδια (επικρατέστερη) ομάδα. Οι ετικέτες του K-Means είναι αυθαίρετες από
    ημέρα σε ημέρα, οπότε το νούμερο αυτό είναι ενδεικτικό. Το αυστηρό μέτρο
    είναι το ARI.
    """
    frame = pd.DataFrame(labels_by_day)
    present = frame.notna().sum(axis=1)
    modal_share = frame.apply(
        lambda row: row.value_counts(normalize=True).iloc[0] if row.notna().any() else np.nan,
        axis=1)
    return pd.DataFrame({'Ημέρες': present, 'Σταθερότητα': modal_share})


def reference_labels(level):
    """Οι ετικέτες της συνολικής ομαδοποίησης, ή None αν δεν έχει τρέξει."""
    folder, name = REFERENCE_FILES[level]
    path = os.path.join(BASE_DIR, 'results', folder, name)
    if not os.path.exists(path):
        return None
    ref = pd.read_csv(path, usecols=['Host', 'Cluster']).set_index('Host')['Cluster']
    return ref[ref >= 0]


def align_day(day_labels, reference):
    """
    Μεταφράζει τις ετικέτες μιας ημέρας στις ετικέτες της αναφοράς.

    Ο K-Means δίνει αυθαίρετη αρίθμηση σε κάθε εκτέλεση: το cluster 1 της
    Τρίτης δεν έχει καμία σχέση με το cluster 1 της Τετάρτης, και χωρίς
    αντιστοίχιση οι ετικέτες δεν συγκρίνονται ούτε σε πίνακα ούτε σε γράφημα.

    Φτιάχνω τον πίνακα επικάλυψης στους κοινούς hosts και λύνω το πρόβλημα
    ανάθεσης (Hungarian algorithm) ώστε να μεγιστοποιηθεί το άθροισμα της
    διαγωνίου. Είναι η βέλτιστη αντιστοίχιση 1-προς-1, όχι greedy: το greedy
    «κλειδώνει» το μεγαλύτερο ζεύγος πρώτο και μπορεί να αναγκάσει τα
    υπόλοιπα σε κακές επιλογές.

    Επιστρέφει (μεταφρασμένες ετικέτες, κοινοί hosts, ποσοστό συμφωνίας,
    η αντιστοίχιση ως κείμενο).
    """
    common = day_labels.dropna().index.intersection(reference.index)
    if len(common) == 0:
        return None, 0, float('nan'), ""

    overlap = pd.crosstab(day_labels.loc[common], reference.loc[common])
    # Οι στήλες συμπληρώνονται με ΟΛΕΣ τις ετικέτες της αναφοράς, ακόμη και
    # όσες δεν εμφανίζονται στους κοινούς hosts της συγκεκριμένης ημέρας.
    # Αλλιώς ο πίνακας βγαίνει με λιγότερες στήλες από γραμμές, μια ομάδα της
    # ημέρας μένει αταίριαστη και παίρνει παρακάτω «εφεδρικό» αριθμό max+1 —
    # ετικέτα που ΔΕΝ υπάρχει στη συνολική ομαδοποίηση. Το αποτέλεσμα είναι
    # ομάδα φάντασμα στα γραφήματα σύνθεσης και στους πίνακες συμφωνίας, ενώ
    # η σωστή απάντηση ήταν να αντιστοιχιστεί στην ομάδα που έλειπε.
    # Αυτό ακριβώς φυλάει το tests/test_pipeline.py::TestAlignment.
    overlap = overlap.reindex(columns=sorted(reference.unique()), fill_value=0)
    rows, cols = linear_sum_assignment(-overlap.to_numpy())
    mapping = {overlap.index[r]: overlap.columns[c] for r, c in zip(rows, cols)}
    matched = sum(overlap.iat[r, c] for r, c in zip(rows, cols))

    # Αν η ημέρα έχει περισσότερες ομάδες από την αναφορά, οι περισσευούμενες
    # παίρνουν καινούριους αριθμούς αντί να μηδενιστούν.
    spare = int(max(mapping.values(), default=-1))
    for label in overlap.index:
        if label not in mapping:
            spare += 1
            mapping[label] = spare

    translated = day_labels.map(mapping).astype('Int8')
    text = ", ".join(f"{a}->{b}" for a, b in sorted(mapping.items()))
    return translated, len(common), matched / len(common), text


def align_all(level, labels_by_day):
    """
    Αντιστοιχίζει όλες τις ημέρες στην ίδια αναφορά και συνοψίζει.

    Επιστρέφει (ετικέτες ανά ημέρα, πίνακας ποιότητας, σύνθεση ανά ημέρα),
    ή None αν λείπει η συνολική ομαδοποίηση.
    """
    reference = reference_labels(level)
    if reference is None or reference.empty:
        print("  Δεν βρέθηκε η συνολική ομαδοποίηση, οπότε δεν γίνεται "
              "αντιστοίχιση ετικετών.")
        return None

    aligned, quality = {}, []
    for day in sorted(labels_by_day):
        labels, common, agreement, mapping = align_day(labels_by_day[day], reference)
        if labels is None:
            continue
        aligned[day] = labels
        quality.append({'Ημέρα': day, 'Κοινοί_hosts': common,
                        'Συμφωνία': agreement, 'Αντιστοίχιση': mapping})

    if not aligned:
        return None

    quality = pd.DataFrame(quality)
    composition = pd.DataFrame(aligned).T.apply(
        lambda row: row.value_counts(), axis=1).fillna(0).astype(int)
    composition.index.name = 'Ημέρα'
    composition.columns = [f'Cluster_{c}' for c in composition.columns]

    print(f"  Αντιστοίχιση ετικετών στη συνολική ομαδοποίηση: "
          f"συμφωνία διάμεσος {quality['Συμφωνία'].median():.3f}, "
          f"εύρος [{quality['Συμφωνία'].min():.3f}, {quality['Συμφωνία'].max():.3f}]")
    return aligned, quality, composition


def analyse(level, labels_by_day, n_clusters):
    title = "Εξωτερικοί hosts" if level == 'macro' else "Εσωτερικοί hosts"
    print(f"\n{'=' * 70}")
    print(f"  ΕΥΣΤΑΘΕΙΑ ΑΝΑ ΗΜΕΡΑ — {title} (k={n_clusters})")
    print(f"{'=' * 70}")

    usable = {d: s for d, s in labels_by_day.items() if s is not None}
    if len(usable) < 2:
        print("  Λιγότερες από δύο ημέρες με δεδομένα. Παράλειψη.")
        return None

    ari = pairwise_ari(usable)
    if ari.empty:
        print("  Δεν βρέθηκαν κοινοί hosts μεταξύ ημερών.")
        return None

    print(f"  Ζεύγη ημερών: {len(ari)} | Κοινοί hosts ανά ζεύγος: "
          f"διάμεσος {ari['Κοινοί_hosts'].median():,.0f}")
    print(f"  ARI: διάμεσος {ari['ARI'].median():.3f} | "
          f"εύρος [{ari['ARI'].min():.3f}, {ari['ARI'].max():.3f}]")

    cons = consistency(usable)
    all_days = cons[cons['Ημέρες'] == len(usable)]
    print(f"  Hosts σε όλες τις {len(usable)} ημέρες: {len(all_days)}")
    if len(all_days):
        print(f"  Από αυτούς, μέση σταθερότητα επικρατέστερης ομάδας: "
              f"{all_days['Σταθερότητα'].mean():.3f}")

    ari.to_csv(os.path.join(output_dir(), f'ari_{level}.csv'),
               index=False, encoding=CSV_ENCODING)
    # Int8 αντί για float: οι ετικέτες είναι ακέραιες, αλλά όταν ένας host
    # λείπει από μια ημέρα η pandas μετατρέπει τη στήλη σε float και γράφει
    # 0.0 / 1.0 αντί για 0 / 1.
    labels = pd.DataFrame(usable).astype('Int8')
    labels.to_csv(os.path.join(output_dir(), f'daily_labels_{level}.csv'),
                  encoding=CSV_ENCODING)
    cons.to_csv(os.path.join(output_dir(), f'consistency_{level}.csv'),
                encoding=CSV_ENCODING)

    # Οι ετικέτες παραπάνω είναι ασύγκριτες μεταξύ ημερών. Οι παρακάτω είναι
    # μεταφρασμένες στην αρίθμηση της συνολικής ομαδοποίησης, άρα συγκρίσιμες.
    outcome = align_all(level, usable)
    if outcome is not None:
        aligned, quality, composition = outcome
        pd.DataFrame(aligned).to_csv(
            os.path.join(output_dir(), f'daily_labels_aligned_{level}.csv'),
            encoding=CSV_ENCODING)
        quality.to_csv(os.path.join(output_dir(), f'alignment_{level}.csv'),
                       index=False, encoding=CSV_ENCODING)
        composition.to_csv(os.path.join(output_dir(), f'composition_{level}.csv'),
                           encoding=CSV_ENCODING)
    return ari


def resolve_k(level, explicit, configured):
    """
    Το k της ημερήσιας ομαδοποίησης, με σειρά προτεραιότητας:
    --k-macro/--k-micro, μετά ό,τι διάλεξε το αντίστοιχο στάδιο macro/micro
    σε αυτή την εκτέλεση, και τελευταίο το config.

    Ο λόγος είναι η αντιστοίχιση ετικετών παρακάτω: συγκρίνει τις ημερήσιες
    ομάδες με τη ΣΥΝΟΛΙΚΗ ομαδοποίηση που έφτιαξε το στάδιο macro/micro. Αν
    εκεί ο χρήστης απαντούσε k=4 στην ερώτηση, εδώ θα τρέχαμε σιωπηλά με το 3
    του config και θα συγκρίναμε τρεις ομάδες με τέσσερις — χωρίς καμία
    ένδειξη ότι τα δύο στάδια μιλούν για διαφορετικά πράγματα.
    """
    if explicit is not None:
        return explicit
    try:
        from src.models.select_k import K_USED
        used = K_USED.get(level, {}).get('k')
    except Exception:
        used = None
    if used is not None and used != configured:
        print(f"    - Το στάδιο {level} έτρεξε με k={used} (αντί για {configured} "
              f"του config). Χρησιμοποιώ το ίδιο, ώστε οι ετικέτες να "
              f"αντιστοιχίζονται.")
    return configured if used is None else used


def run(k_macro=None, k_micro=None):
    print("\n--- Έναρξη: Ανάλυση Ευστάθειας ανά Ημέρα ---")
    bidir_files = sorted(glob.glob(os.path.join(PATHS['bidirectional_data'], "bidir_*.csv")))
    if len(bidir_files) < 2:
        print("Προειδοποίηση: Χρειάζονται τουλάχιστον δύο ημέρες. Παράλειψη.")
        return

    k_macro = resolve_k('macro', k_macro, MACRO_CLUSTERS)
    k_micro = resolve_k('micro', k_micro, MICRO_CLUSTERS)
    print(f" -> {len(bidir_files)} ημέρες | k macro={k_macro}, micro={k_micro}")

    macro_labels, micro_labels = {}, {}
    for path in bidir_files:
        day = day_of(path)
        print(f"    - Ομαδοποίηση ημέρας {day}...")
        profiles = build_profiles([path])
        internal, external = split_internal_external(profiles.reset_index())
        for frame, store, k in ((external, macro_labels, k_macro),
                                (internal, micro_labels, k_micro)):
            labels = cluster_day(frame.set_index('Host'), k) if not frame.empty else None
            store[day] = labels

    analyse('macro', macro_labels, k_macro)
    analyse('micro', micro_labels, k_micro)

    print(f"\n -> Τα αποτελέσματα αποθηκεύτηκαν στο: {output_dir()}")
    print("--- Ολοκληρώθηκε η Ανάλυση Ευστάθειας ---")


if __name__ == "__main__":
    run()
