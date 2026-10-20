"""
Καταγραφή εκτέλεσης σε results/run_manifest.json.

Το pipeline είναι ντετερμινιστικό (σταθερό RANDOM_STATE), αλλά αυτό από μόνο
του δεν κάνει ένα αποτέλεσμα αναπαραγώγιμο. Για να ξαναβγεί ο ίδιος αριθμός
χρειάζονται τρία πράγματα: οι ίδιες παράμετροι, τα ίδια δεδομένα εισόδου και
οι ίδιες εκδόσεις βιβλιοθηκών. Το UMAP ειδικά αλλάζει έξοδο μεταξύ εκδόσεων.

Αυτό το module γράφει αυτά τα τρία σε ένα αρχείο μετά από κάθε εκτέλεση, ώστε
κάθε γράφημα και κάθε πίνακας της αναφοράς να μπορεί να συνδεθεί με τη
συγκεκριμένη εκτέλεση που τον παρήγαγε.

Σε ποια αρχεία υπολογίζεται SHA-256 και γιατί όχι σε όλα:
  - Raw data: μόνο όνομα, μέγεθος και ημερομηνία. Είναι 3,6 GB και το SHA-256
    τους θα πρόσθετε μισό λεπτό σε κάθε εκτέλεση. Το τρίπτυχο αρκεί για να
    δεις ότι μιλάμε για τα ίδια αρχεία.
  - Ενδιάμεσα (2_processed, 3_bidirectional): όνομα και μέγεθος. Παράγονται
    ντετερμινιστικά από τα raw, άρα δεν προσθέτουν πληροφορία.
  - features_global.csv και όλα τα CSV του results/: πλήρες SHA-256. Είναι
    μικρά και είναι αυτά που καταλήγουν στην αναφορά.

Το THESIS_SALT_KEY δεν γράφεται εδώ. Το manifest προορίζεται
να μοιραστεί μαζί με τον κώδικα και τα αποτελέσματα.
"""
import glob
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata

MANIFEST_NAME = 'run_manifest.json'

# Οι βιβλιοθήκες που επηρεάζουν τα αποτελέσματα. Το numba/llvmlite/pynndescent
# μπαίνουν επειδή από αυτά εξαρτάται η έξοδος του UMAP.
TRACKED_PACKAGES = (
    'numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib', 'seaborn',
    'umap-learn', 'numba', 'llvmlite', 'pynndescent', 'joblib', 'python-dotenv',
)

HASH_CHUNK = 1 << 20  # 1 MiB ανά ανάγνωση: δεν φορτώνει ολόκληρο αρχείο στη μνήμη


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path, with_hash=False):
    """Ταυτότητα ενός αρχείου: μέγεθος, ημερομηνία και προαιρετικά SHA-256."""
    stat = os.stat(path)
    entry = {
        'bytes': stat.st_size,
        'modified': datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                            .isoformat(timespec='seconds'),
    }
    if with_hash:
        entry['sha256'] = sha256(path)
    return entry


def describe_folder(folder, pattern='*.csv', with_hash=False):
    """{όνομα αρχείου: ταυτότητα} για ένα φάκελο, ταξινομημένο."""
    if not os.path.isdir(folder):
        return {}
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    return {os.path.basename(p): describe(p, with_hash) for p in files}


def packages():
    """Εγκατεστημένες εκδόσεις. Ό,τι λείπει γράφεται ως None αντί να σκάσει."""
    found = {}
    for name in TRACKED_PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def k_actually_used():
    """
    Το k που δόθηκε ΣΤΗΝ ΠΡΑΞΗ στον K-Means, και από πού ήρθε.

    Τα MACRO_CLUSTERS / MICRO_CLUSTERS του config είναι μόνο η προεπιλογή. Το
    πραγματικό k μπορεί να έρθει από το --k-macro/--k-micro ή από την απάντηση
    του χρήστη στην ερώτηση των σταδίων macro/micro. Χωρίς αυτό, το manifest
    θα κατέγραφε την προεπιλογή ως «την παράμετρο της εκτέλεσης» ακόμη κι όταν
    δεν ήταν.

    Άδειο όταν η εκτέλεση δεν περιλάμβανε στάδιο που επιλέγει k.
    """
    try:
        from src.models.select_k import K_USED
    except Exception:
        return {}
    return {level: dict(entry) for level, entry in K_USED.items()}


def parameters(config):
    """Οι παράμετροι που καθορίζουν το αποτέλεσμα. Χωρίς το κλειδί."""
    return {
        'RANDOM_STATE': config.RANDOM_STATE,
        'CONTAMINATION_RATE': config.CONTAMINATION_RATE,
        'MACRO_CLUSTERS': config.MACRO_CLUSTERS,
        'MICRO_CLUSTERS': config.MICRO_CLUSTERS,
        'k_used': k_actually_used(),
        'PARTIAL_DAYS': list(config.PARTIAL_DAYS),
        'INTERNAL_SUBNET': config.INTERNAL_SUBNET,
        'BUSINESS_HOURS': list(config.BUSINESS_HOURS),
        'WELL_KNOWN_PORT_MAX': config.WELL_KNOWN_PORT_MAX,
        'P2P_PORTS': sorted(config.P2P_PORTS),
        'FEATURE_COLUMNS': list(config.FEATURE_COLUMNS),
        'LOG_FEATURES': list(config.LOG_FEATURES),
        'LINEAR_FEATURES': list(config.LINEAR_FEATURES),
        'RATIO_FEATURES': list(config.RATIO_FEATURES),
        'DBSCAN': {
            'eps': config.DBSCAN_EPS,
            'eps_percentile': config.DBSCAN_EPS_PERCENTILE,
            'min_samples': config.DBSCAN_MIN_SAMPLES,
            'min_samples_fraction': config.DBSCAN_MIN_SAMPLES_FRACTION,
            'min_samples_max': config.DBSCAN_MIN_SAMPLES_MAX,
            'max_edges': config.DBSCAN_MAX_EDGES,
        },
        'CHUNK_SIZE': config.CHUNK_SIZE,
    }


def write(steps, succeeded, seconds, options=None):
    """
    Γράφει το manifest. Οι αποτυχίες καταγράφονται κι αυτές: μια εκτέλεση που
    διακόπηκε στη μέση αφήνει ανάμεικτα αποτελέσματα και αυτό πρέπει να φαίνεται.
    """
    from src import config  # εδώ και όχι στην κορυφή: το main.py φορτώνει το config αργά

    base = config.BASE_DIR
    paths = config.PATHS
    results_dir = os.path.join(base, 'results')
    os.makedirs(results_dir, exist_ok=True)

    manifest = {
        'generated': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'run': {
            'steps': list(steps),
            'succeeded': bool(succeeded),
            'seconds': round(float(seconds), 1),
            'cli_options': {k: v for k, v in (options or {}).items() if v is not None},
        },
        'environment': {
            'python': sys.version.split()[0],
            'python_full': sys.version,
            'platform': platform.platform(),
            'machine': platform.machine(),
            'packages': packages(),
        },
        'parameters': parameters(config),
        'inputs': {
            'raw': describe_folder(paths['raw_data'], '*.csv'),
            'processed': describe_folder(paths['processed_data'], 'clean_*.csv'),
            'bidirectional': describe_folder(paths['bidirectional_data'], '*.csv'),
            'profiles': describe_folder(paths['features'], 'features_*.csv', with_hash=True),
        },
        'outputs': {
            name: describe_folder(os.path.join(results_dir, name), '*.csv', with_hash=True)
            for name in ('macro_clustering', 'micro_clustering', 'k_selection',
                         'sensitivity', 'stability', 'peers', 'hourly', 'ports')
        },
    }
    manifest['outputs']['graphs'] = describe_folder(
        os.path.join(results_dir, 'graphs'), '*.png', with_hash=True)

    path = os.path.join(results_dir, MANIFEST_NAME)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path
