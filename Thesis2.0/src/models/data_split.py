"""
Ο διαχωρισμός των δεδομένων, κοινός για τα μοντέλα και για την επιλογή του k.

Το έβγαλα σε ξεχωριστό module γιατί αλλιώς η εξάρτηση γινόταν κυκλική: η
επιλογή του k πρέπει να δει ακριβώς τα δεδομένα που θα δει ο K-Means
(εξωτερικοί hosts χωρίς τις ανωμαλίες), αλλά αυτά παράγονταν μέσα στη
μακροσκοπική φάση, δηλαδή μετά. Τώρα ο διαχωρισμός ορίζεται μία φορά και τον
καλούν και τα δύο.

Οι συναρτήσεις δέχονται πλέον το σύνολο hashes ως παράμετρο, με προεπιλογή το
εσωτερικό subnet. Έτσι ο ίδιος κώδικας εξυπηρετεί και τα υπόλοιπα ιδιωτικά
υποδίκτυα (δες populations.py) χωρίς αντίγραφο, και χωρίς να αλλάξει τίποτα
για όποιον τις καλεί όπως πριν.
"""
from functools import lru_cache

from sklearn.ensemble import IsolationForest

from src.config import CONTAMINATION_RATE, RANDOM_STATE
from src.utils.anonymize import get_internal_hashes, get_network_hashes
from src.utils.scaling import scale_features


@lru_cache(maxsize=1)
def private_hosts():
    """
    Οι ιδιωτικές διευθύνσεις εκτός των δικών μας υποδικτύων.

    Διαβάζεται από τον κατάλογο που γράφει το βήμα 'privatescan'. Με lru_cache
    επειδή το daily_stability χωρίζει οκτώ φορές, μία ανά ημέρα.

    Αν ο κατάλογος λείπει, το σύνολο είναι άδειο και το λέει δυνατά: χωρίς
    αυτόν οι ιδιωτικές διευθύνσεις ξαναμπαίνουν στους «εξωτερικούς hosts του
    διαδικτύου», που είναι το λάθος που το βήμα διορθώνει.
    """
    from src.utils.private_scan import OUTPUT, others
    found = others()
    if not found:
        print(f"    [!] Δεν βρέθηκε ο κατάλογος ιδιωτικών ({OUTPUT}).")
        print("        Οι ιδιωτικές διευθύνσεις άλλων /24 θα μετρηθούν ως "
              "εξωτερικές.")
        print("        Τρέξε:  python main.py --step privatescan")
    return found


def split_internal_external(df, internal_hashes=None, network_hashes=None,
                            other_private=None):
    """
    (εσωτερικοί hosts, εξωτερικοί hosts).

    Τα δύο σύνολα ΔΕΝ είναι συμπληρωματικά, και αυτό είναι σκόπιμο. Εσωτερικοί
    είναι οι hosts των δικών μας υποδικτύων (οκτέτα 1-254). Εξωτερικοί είναι
    ό,τι δεν ανήκει καθόλου στα δικά μας υποδίκτυα (0-255) ΚΑΙ δεν είναι
    ιδιωτική διεύθυνση RFC1918. Στη διαφορά πέφτουν δύο κατηγορίες:

      - οι διευθύνσεις δικτύου και εκπομπής των δικών μας υποδικτύων: δεν
        είναι μηχανές για να ομαδοποιηθούν, ούτε hosts του διαδικτύου για να
        μετρηθούν ως ανωμαλίες. Όσο ήταν συμπληρωματικά, η 192.168.7.255
        εμφανιζόταν στις ανωμαλίες του Isolation Forest.

      - οι ιδιωτικές διευθύνσεις άλλων /24 (85 στα δεδομένα, από το βήμα
        'privatescan'). Μια διεύθυνση RFC1918 δεν είναι host του διαδικτύου με
        καμία ανάγνωση, και 20 από αυτές έβγαιναν στις ανωμαλίες ως «ύποπτοι
        εξωτερικοί hosts» ενώ ήταν υποδομή: OpenFlow, DNS, NTP, Kubernetes.
        Δεν μπαίνουν ΟΥΤΕ στους εσωτερικούς: δεν υπάρχει ένδειξη ότι ανήκουν
        στη σχολή, και είναι σκόρπιες σε δεκαπέντε διαφορετικά /24.

    Με ρητό internal_hashes και χωρίς network_hashes τα δύο ταυτίζονται, ώστε
    όποιος καλεί με δικό του σύνολο να παίρνει τον απλό συμπληρωματικό
    διαχωρισμό που περιμένει — και τότε δεν αφαιρούνται ιδιωτικές, γιατί ο
    καλών έχει ορίσει ο ίδιος τι θεωρεί δικό του.
    """
    explicit = internal_hashes is not None
    if internal_hashes is None:
        internal_hashes = get_internal_hashes()
        if network_hashes is None:
            network_hashes = get_network_hashes()
    if network_hashes is None:
        network_hashes = internal_hashes
    if other_private is None:
        other_private = frozenset() if explicit else private_hosts()

    inside = df['Host'].isin(internal_hashes)
    ours = df['Host'].isin(network_hashes)
    elsewhere = df['Host'].isin(other_private)
    return df[inside].copy(), df[~ours & ~elsewhere].copy()


def remove_anomalies(df_external):
    """
    Isolation Forest στους εξωτερικούς hosts.

    Γυρίζει (καθαροί, ανωμαλίες). Κρατώ τη στήλη 'Anomaly' (1 / -1) όπως ήταν,
    γιατί τη περιμένει έτσι η μακροσκοπική φάση.
    """
    df = df_external.copy()
    X = scale_features(df)
    iso = IsolationForest(contamination=CONTAMINATION_RATE, random_state=RANDOM_STATE)
    df['Anomaly'] = iso.fit_predict(X)
    return df[df['Anomaly'] == 1].copy(), df[df['Anomaly'] == -1].copy()


def macro_dataset(df, internal_hashes=None):
    """Τα δεδομένα που βλέπει ο K-Means της μακροσκοπικής ανάλυσης."""
    _, external = split_internal_external(df, internal_hashes)
    if external.empty:
        return external, external
    return remove_anomalies(external)


def micro_dataset(df, internal_hashes=None):
    """Τα δεδομένα που βλέπει ο K-Means της μικροσκοπικής ανάλυσης."""
    internal, _ = split_internal_external(df, internal_hashes)
    return internal


def population_dataset(df, key):
    """
    Οι hosts ενός πληθυσμού, από το κλειδί του.

    Η εισαγωγή είναι μέσα στη συνάρτηση επειδή το populations.py δεν πρέπει να
    φορτώνεται όταν κάποιος χρειάζεται μόνο τον κλασικό διαχωρισμό: το
    populations υπολογίζει hashes, δηλαδή θέλει το κλειδί από το .env.
    """
    from src.models.populations import hashes_of
    return df[df['Host'].isin(hashes_of(key))].copy()
