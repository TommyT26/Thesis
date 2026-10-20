"""
Επιλογή του αριθμού clusters k για τον K-Means, με πολλαπλούς δείκτες εγκυρότητας.

Το clustering είναι μη-επιβλεπόμενο: δεν υπάρχει ground truth, άρα ούτε
απόδειξη ενός σωστού k. Αυτό που μπορώ να κάνω είναι να υπολογίσω ανεξάρτητους
δείκτες και να δω σε ποιες τιμές συγκλίνουν.

Κριτήρια:
  1. Elbow / inertia (WCSS)     - μέγιστη καμπυλότητα (απόσταση από τη χορδή)
  2. Silhouette                 - συνοχή vs διαχωρισμός, ΜΕΓΑΛΥΤΕΡΟ = καλύτερο
  3. Calinski-Harabasz          - λόγος διασποράς between/within, ΜΕΓΑΛΥΤΕΡΟ
  4. Davies-Bouldin             - ομοιότητα με τον πλησιέστερο cluster, ΜΙΚΡΟΤΕΡΟ
  5. Gap statistic (Tibshirani) - σύγκριση με ομοιόμορφη κατανομή αναφοράς
  6. Ευστάθεια (bootstrap ARI)  - αναπαραγωγιμότητα σε ανεξάρτητα υποδείγματα

Το module κάνει δύο πράγματα:
  - run()      : υπολογίζει τους δείκτες και τους αποθηκεύει
  - choose_k() : δείχνει τον πίνακα και ρωτά για το k πριν τρέξει ο K-Means
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score)

from src.config import PATHS, BASE_DIR, RANDOM_STATE, MACRO_CLUSTERS, MICRO_CLUSTERS
from src.utils.scaling import scale_features
from src.models.data_split import macro_dataset, micro_dataset

MACRO_K_RANGE = range(2, 16)
MICRO_K_RANGE = range(2, 11)
SILHOUETTE_SAMPLE = 5000    # το silhouette είναι O(n^2): δειγματοληψία στο macro
SILHOUETTE_REPEATS = 3
GAP_REFS = 10
GAP_SAMPLE = 20000
STABILITY_ROUNDS = 10
STABILITY_FRACTION = 0.6

LEVELS = {
    'macro': ('Μακροσκοπική ανάλυση (εξωτερικοί hosts)', MACRO_CLUSTERS),
    'micro': ('Μικροσκοπική ανάλυση (εσωτερικοί hosts)', MICRO_CLUSTERS),
}


def output_dir():
    path = os.path.join(BASE_DIR, 'results', 'k_selection')
    os.makedirs(path, exist_ok=True)
    return path


# ==========================================================================
#  Υπολογισμός δεικτών
# ==========================================================================
def fit(X, k, n_init=10, seed=RANDOM_STATE):
    return KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(X)


def knee_point(ks, values):
    """Σημείο μέγιστης απόστασης από τη χορδή πρώτου-τελευταίου σημείου (Kneedle)."""
    x, y = np.asarray(ks, float), np.asarray(values, float)
    x = (x - x.min()) / (np.ptp(x) or 1)
    y = (y - y.min()) / (np.ptp(y) or 1)
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    num = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
    return int(np.asarray(ks)[int(np.argmax(num / np.hypot(y1 - y0, x1 - x0)))])


def silhouette_sampled(X, labels, sample=SILHOUETTE_SAMPLE, repeats=SILHOUETTE_REPEATS):
    scores = []
    for r in range(repeats):
        rng = np.random.default_rng(RANDOM_STATE + r)
        idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
        if len(np.unique(labels[idx])) > 1:
            scores.append(silhouette_score(X[idx], labels[idx]))
    return (float(np.mean(scores)), float(np.std(scores))) if scores else (np.nan, np.nan)


def gap_statistic(X, ks, n_refs=GAP_REFS, sample=GAP_SAMPLE, seed=RANDOM_STATE):
    """
    Tibshirani et al. (2001). Η κατανομή αναφοράς είναι ομοιόμορφη μέσα στο
    κουτί των κύριων αξόνων των δεδομένων, ώστε να ακολουθεί το σχήμα τους.
    """
    rng = np.random.default_rng(seed)
    if len(X) > sample:
        X = X[rng.choice(len(X), size=sample, replace=False)]

    pca = PCA(n_components=X.shape[1], random_state=seed).fit(X)
    Xp = pca.transform(X)
    lo, hi = Xp.min(axis=0), Xp.max(axis=0)

    gaps, sks = [], []
    for k in ks:
        logw_real = np.log(fit(X, k, n_init=5).inertia_)
        logw_refs = np.array([
            np.log(fit(pca.inverse_transform(rng.uniform(lo, hi, size=Xp.shape)),
                       k, n_init=3, seed=seed + b).inertia_)
            for b in range(n_refs)
        ])
        gaps.append(logw_refs.mean() - logw_real)
        sks.append(logw_refs.std() * np.sqrt(1 + 1.0 / n_refs))
    return np.asarray(gaps), np.asarray(sks)


def gap_best_k(df):
    """Κανόνας Tibshirani: το μικρότερο k με Gap(k) >= Gap(k+1) - s(k+1)."""
    g, s, ks = df['Gap'].to_numpy(), df['Gap_s'].to_numpy(), df['k'].to_numpy()
    return next((int(ks[i]) for i in range(len(ks) - 1)
                 if g[i] >= g[i + 1] - s[i + 1]), None)


def stability(X, k, n_rounds=STABILITY_ROUNDS, frac=STABILITY_FRACTION, seed=RANDOM_STATE):
    """Δύο ανεξάρτητα υποδείγματα -> Adjusted Rand Index στην τομή τους."""
    scores, n = [], len(X)
    for r in range(n_rounds):
        rng = np.random.default_rng(seed + r * 101)
        a = rng.choice(n, size=int(frac * n), replace=False)
        b = rng.choice(n, size=int(frac * n), replace=False)
        common = np.intersect1d(a, b)
        if len(common) < k * 2:
            continue
        la = fit(X[a], k, n_init=3, seed=seed + r).predict(X[common])
        lb = fit(X[b], k, n_init=3, seed=seed + r + 50).predict(X[common])
        scores.append(adjusted_rand_score(la, lb))
    return (float(np.mean(scores)), float(np.std(scores))) if scores else (np.nan, np.nan)


def compute_indices(X, ks, gap_refs=GAP_REFS, stab_rounds=STABILITY_ROUNDS):
    ks = [k for k in ks if k < len(X)]
    rows = []
    for k in ks:
        km = fit(X, k)
        sil, sil_sd = silhouette_sampled(X, km.labels_)
        rows.append({
            'k': k, 'Inertia': km.inertia_,
            'Silhouette': sil, 'Sil_sd': sil_sd,
            'Calinski_Harabasz': calinski_harabasz_score(X, km.labels_),
            'Davies_Bouldin': davies_bouldin_score(X, km.labels_),
        })
        print(f"      k={k} ολοκληρώθηκε", end='\r')
    df = pd.DataFrame(rows)

    print("      Gap statistic...        ", end='\r')
    df['Gap'], df['Gap_s'] = gap_statistic(X, ks, n_refs=gap_refs)

    print("      Ευστάθεια (bootstrap ARI)...", end='\r')
    stabs = [stability(X, k, n_rounds=stab_rounds) for k in ks]
    df['Stability_ARI'] = [s[0] for s in stabs]
    df['Stab_sd'] = [s[1] for s in stabs]
    print(" " * 50, end='\r')
    return df


def votes(df):
    """Η πρόταση κάθε κριτηρίου ξεχωριστά."""
    return {
        'Elbow': knee_point(df['k'], df['Inertia']),
        'Silhouette': int(df.loc[df['Silhouette'].idxmax(), 'k']),
        'Calinski-Harabasz': int(df.loc[df['Calinski_Harabasz'].idxmax(), 'k']),
        'Davies-Bouldin': int(df.loc[df['Davies_Bouldin'].idxmin(), 'k']),
        'Gap statistic': gap_best_k(df),
        'Ευστάθεια ARI': int(df.loc[df['Stability_ARI'].idxmax(), 'k']),
    }


# ==========================================================================
#  Εμφάνιση και επιλογή
# ==========================================================================
def table_path(level):
    return os.path.join(output_dir(), f'k_selection_{level}.csv')


def load_k_table(level):
    path = table_path(level)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return df if {'k', 'Silhouette'}.issubset(df.columns) and not df.empty else None


def print_k_table(level, df):
    """Πίνακας δεικτών με αστερίσκο στη βέλτιστη τιμή κάθε στήλης."""
    title, configured = LEVELS[level]
    v = votes(df)
    best = {
        'Silhouette': v['Silhouette'],
        'Calinski_Harabasz': v['Calinski-Harabasz'],
        'Davies_Bouldin': v['Davies-Bouldin'],
        'Gap': v['Gap statistic'],
        'Stability_ARI': v['Ευστάθεια ARI'],
    }

    print("\n" + "=" * 72)
    print(f"  ΔΕΙΚΤΕΣ ΕΓΚΥΡΟΤΗΤΑΣ — {title}")
    print("=" * 72)
    print(f"  {'k':>3}  {'Silhouette':>11} {'Calinski-H.':>13} {'Davies-B.':>11} "
          f"{'Gap':>8} {'Ευστάθεια':>10}")
    print("  " + "-" * 68)
    for _, row in df.iterrows():
        k = int(row['k'])
        cells = []
        for col, width, fmt in [('Silhouette', 11, '.4f'), ('Calinski_Harabasz', 13, ',.1f'),
                                ('Davies_Bouldin', 11, '.4f'), ('Gap', 8, '.4f'),
                                ('Stability_ARI', 10, '.4f')]:
            mark = '*' if best.get(col) == k else ' '
            cells.append(f"{format(row[col], fmt):>{width - 1}}{mark}")
        print(f"  {k:>3}  " + " ".join(cells))
    print("  " + "-" * 68)
    print("  (* = βέλτιστη τιμή της στήλης· μεγαλύτερο καλύτερο, εκτός Davies-Bouldin)")

    print("\n  Προτάσεις κριτηρίων:")
    items = [f"{name} → k={k}" for name, k in v.items() if k is not None]
    for i in range(0, len(items), 2):
        print("     " + "   |   ".join(items[i:i + 2]))
    if v['Gap statistic'] is None:
        print("     Gap statistic → δεν σταθεροποιείται στο εύρος που δοκιμάστηκε")
    print(f"\n  Τιμή στο config.py: k = {configured}")


def ask(prompt):
    """input() που δεν σκάει όταν κλείσει το stdin."""
    try:
        return input(prompt)
    except EOFError:
        return None


# Τι k χρησιμοποιήθηκε ΣΤΗΝ ΠΡΑΞΗ σε αυτή την εκτέλεση, και από πού ήρθε.
# Το manifest κατέγραφε μόνο το MACRO_CLUSTERS/MICRO_CLUSTERS του config.py,
# αλλά το k μπορεί να έρθει από τρία σημεία: το config, το --k-macro/--k-micro,
# ή την απάντηση του χρήστη στην ερώτηση. Αν κάποιος απαντούσε 4 ενώ το config
# λέει 3, το manifest θα έγραφε 3 — δηλαδή το αρχείο που υπάρχει ειδικά για να
# τεκμηριώνει την εκτέλεση θα έλεγε ψέματα γι' αυτήν. Εδώ κρατάω την αλήθεια.
K_USED = {}


def choose_k(level, default=None):
    """
    Το k που θα χρησιμοποιήσει ο K-Means.

    Αν υπάρχουν αποτελέσματα από το στάδιο 'kselect' και τρέχουμε σε τερματικό,
    δείχνει τον πίνακα και ρωτά. Σε κάθε άλλη περίπτωση (cron, nohup,
    ανακατεύθυνση εξόδου, ή αν λείπουν τα αποτελέσματα) γυρίζει την προεπιλογή
    χωρίς να περιμένει απάντηση.

    Ό,τι επιστραφεί καταγράφεται στο K_USED μαζί με την προέλευσή του, ώστε να
    μπει στο run_manifest.json.
    """
    title, configured = LEVELS[level]
    explicit = default is not None
    default = configured if default is None else default

    def remember(value, source):
        K_USED[level] = {'k': int(value), 'source': source}
        return value

    df = load_k_table(level)
    if df is None:
        print(f"    - Δεν βρέθηκαν δείκτες εγκυρότητας· χρήση k={default} από το config.py.")
        print("      (Για τεκμηριωμένη επιλογή τρέξε πρώτα:  python main.py --step kselect)")
        return remember(default, 'cli' if explicit else 'config')

    if not sys.stdin.isatty():
        print(f"    - Μη διαδραστική εκτέλεση· χρήση k={default}.")
        return remember(default, 'cli' if explicit else 'config')

    print_k_table(level, df)
    valid = set(df['k'].astype(int))
    while True:
        answer = ask(f"\n  Επίλεξε k για {level} [Enter = {default}]: ")
        if answer is None:
            print(f"  -> Χωρίς είσοδο. Χρήση k={default}.")
            return remember(default, 'cli' if explicit else 'config')
        answer = answer.strip()
        if not answer:
            return remember(default, 'cli' if explicit else 'config')
        if not answer.isdigit() or int(answer) < 2:
            print("  [!] Δώσε ακέραιο >= 2.")
            continue
        chosen = int(answer)
        if chosen not in valid:
            confirm = ask(f"  [!] Το k={chosen} είναι εκτός του εύρους που αξιολογήθηκε "
                          f"({min(valid)}-{max(valid)}). Συνέχεια; (y/n): ")
            if not confirm or confirm.strip().lower() != 'y':
                continue
        return remember(chosen, 'interactive')


# ==========================================================================
#  Στάδιο pipeline
# ==========================================================================
def analyse(level, X, ks, **kwargs):
    title, _ = LEVELS[level]
    print(f"\n -> {title}: n={len(X)}, d={X.shape[1]}")
    if len(X) <= 2:
        print("    - Πολύ λίγα σημεία για αξιολόγηση. Παράλειψη.")
        return None
    df = compute_indices(X, ks, **kwargs)
    df.to_csv(table_path(level), index=False)
    print_k_table(level, df)
    return df


def run():
    print("\n--- Έναρξη: Επιλογή αριθμού clusters k (δείκτες εγκυρότητας) ---")
    feature_files = glob.glob(os.path.join(PATHS['features'], "features_*.csv"))
    if not feature_files:
        print(f"[!] Δεν βρέθηκαν αρχεία στον φάκελο {PATHS['features']}")
        return

    df = pd.read_csv(feature_files[0])
    if len(feature_files) > 1:
        print(f" -> Βρέθηκαν {len(feature_files)} αρχεία· χρήση του "
              f"{os.path.basename(feature_files[0])}")

    clean_macro, _ = macro_dataset(df)
    if not clean_macro.empty:
        analyse('macro', scale_features(clean_macro), MACRO_K_RANGE)

    internal = micro_dataset(df)
    if not internal.empty:
        analyse('micro', scale_features(internal), MICRO_K_RANGE,
                gap_refs=20, stab_rounds=20)

    # Έλεγχος ευρωστίας: οι ίδιοι δείκτες χωρίς τα ταυτόσημα προφίλ. Αν το k
    # αλλάζει δραστικά, τότε η δομή που βρήκα οφείλεται στη συχνότητα με την
    # οποία επαναλαμβάνονται τα ίδια προφίλ, όχι στις συμπεριφορές.
    if not clean_macro.empty:
        X = scale_features(clean_macro)
        U = np.unique(np.round(X, 6), axis=0)
        print(f"\n -> Έλεγχος ευρωστίας στα μοναδικά προφίλ: n={len(U)} "
              f"({100 * len(U) / len(X):.1f}% των hosts)")
        compute_indices(U, MACRO_K_RANGE).to_csv(
            os.path.join(output_dir(), 'k_selection_macro_dedup.csv'), index=False)

    print(f"\n -> Τα αποτελέσματα αποθηκεύτηκαν στο: {output_dir()}")
    print("--- Ολοκληρώθηκε η επιλογή k ---")


if __name__ == "__main__":
    run()
