import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from src.config import (
    DBSCAN_EPS, DBSCAN_EPS_PERCENTILE, DBSCAN_MIN_SAMPLES,
    DBSCAN_MIN_SAMPLES_FRACTION, DBSCAN_MIN_SAMPLES_MAX, DBSCAN_MAX_EDGES,
)

# Δεκαδικά ψηφία για τον εντοπισμό ταυτόσημων (κλιμακωμένων) προφίλ.
DEDUP_DECIMALS = 6
# Δείγμα σημείων για την εκτίμηση του μεγέθους του eps-γράφου.
EDGE_SAMPLE = 2000
# Συρρίκνωση του eps όταν ο γράφος δεν χωρά στο budget μνήμης.
EPS_SHRINK_FACTOR = 0.85
MAX_SHRINK_STEPS = 15


def resolve_min_samples(n):
    """
    min_samples ανάλογα με το μέγεθος του συνόλου.

    Σταθερή τιμή δεν δουλεύει εδώ, γιατί η ίδια πρέπει να εξυπηρετήσει και το
    macro (~121.000 εξωτερικοί hosts) και το micro (~64 εσωτερικοί). Με
    min_samples=5 το macro έβγαζε ~660 μικρο-ομάδες που δεν ερμηνεύονται· ως
    ποσοστό του n βγαίνουν λίγες και μεγάλες.
    """
    scaled = round(DBSCAN_MIN_SAMPLES_FRACTION * n)
    return int(min(max(scaled, DBSCAN_MIN_SAMPLES), DBSCAN_MIN_SAMPLES_MAX))


def deduplicate(X, decimals=DEDUP_DECIMALS):
    """
    Συμπτύσσει τα ταυτόσημα προφίλ σε ένα σημείο, με βάρος το πλήθος των διδύμων.

    Γυρίζει (μοναδικά σημεία, index αντιστοίχισης, βάρη). Ο DBSCAN με
    sample_weight μετράει το βάρος αντί για το πλήθος σημείων στη γειτονιά,
    οπότε το αποτέλεσμα είναι ισοδύναμο με εκτέλεση στο πλήρες σύνολο. Απλώς
    χωράει στη μνήμη.
    """
    uniq, inverse, counts = np.unique(
        np.round(X, decimals), axis=0, return_inverse=True, return_counts=True)
    return uniq, np.ravel(inverse), counts.astype(float)


def estimate_eps(U, min_samples, percentile=DBSCAN_EPS_PERCENTILE):
    """
    k-distance heuristic, πάνω στα μοναδικά προφίλ.

    Στα πλήρη δεδομένα το heuristic καταρρέει: όταν ένας host έχει δεκάδες
    χιλιάδες δίδυμα, η απόσταση προς τον k-οστό γείτονα είναι 0 και το
    ποσοστημόριο δίνει eps ~ 0. Μετά το dedup η καμπύλη δείχνει πραγματικές
    αποστάσεις μεταξύ διαφορετικών συμπεριφορών.
    """
    n_neighbors = min(min_samples + 1, len(U))  # +1: ο πρώτος γείτονας είναι το ίδιο το σημείο
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(U).kneighbors(U)
    return max(float(np.percentile(distances[:, -1], percentile)), 1e-3)


def estimate_edges(U, eps, sample=EDGE_SAMPLE, seed=42):
    """Εκτίμηση του πλήθους ακμών του eps-γράφου (μέσος βαθμός x πλήθος σημείων)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(U), size=min(sample, len(U)), replace=False)
    neighbors = NearestNeighbors(radius=eps).fit(U).radius_neighbors(
        U[idx], return_distance=False)
    mean_degree = float(np.mean([len(a) for a in neighbors]))
    return mean_degree * len(U), mean_degree


def fit_dbscan(X, name=""):
    """
    Επιστρέφει (labels, eps). Label -1 = noise (host που δεν ανήκει σε πυκνή ομάδα).
    Αν τα δεδομένα είναι ελάχιστα, επιστρέφει όλα -1.
    """
    n = len(X)
    min_samples = resolve_min_samples(n)
    tag = f" {name}" if name else ""

    if n <= min_samples:
        print(f"    - Προειδοποίηση: Μόνο {n} σημεία (min_samples={min_samples}). Παράλειψη DBSCAN.")
        return np.full(n, -1), None

    # 1. Dedup: το χρειάζομαι και για τη μνήμη και για τη σωστή επιλογή eps.
    U, inverse, weights = deduplicate(X)
    print(f"    - DBSCAN{tag}: {n} hosts -> {len(U)} μοναδικά προφίλ "
          f"({100 * len(U) / n:.1f}%) | min_samples={min_samples}")
    if len(U) < 2:
        print("    - Προειδοποίηση: Όλα τα προφίλ είναι ταυτόσημα. Παράλειψη DBSCAN.")
        return np.full(n, -1), None

    # 2. Επιλογή eps.
    eps = DBSCAN_EPS if DBSCAN_EPS is not None else estimate_eps(U, min_samples)

    # 3. Φραγμή μνήμης: ο sklearn υλοποιεί ολόκληρο τον eps-γράφο στη μνήμη,
    #    οπότε ελέγχω το μέγεθός του πριν το fit.
    for _ in range(MAX_SHRINK_STEPS):
        edges, degree = estimate_edges(U, eps)
        if edges <= DBSCAN_MAX_EDGES:
            break
        print(f"      ! eps={eps:.3f} -> ~{edges:.3g} ακμές (μέσος βαθμός {degree:.0f}), "
              f"πάνω από το όριο {DBSCAN_MAX_EDGES:.3g}. Μείωση eps.")
        eps *= EPS_SHRINK_FACTOR
    else:
        print("      ! Δεν βρέθηκε eps εντός του budget μνήμης. Όλα τα σημεία -> noise.")
        return np.full(n, -1), None

    # 4. DBSCAN στα μοναδικά προφίλ με βάρη, και μετά πίσω στους hosts.
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(U, sample_weight=weights)
    labels_full = labels[inverse]

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels_full == -1).sum())
    print(f"      eps={eps:.3f} | ~{edges:.3g} ακμές | clusters={n_clusters} | "
          f"noise={n_noise} hosts ({100 * n_noise / n:.1f}%)")
    return labels_full, eps
