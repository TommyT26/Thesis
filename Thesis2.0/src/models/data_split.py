"""
Ο διαχωρισμός των δεδομένων, κοινός για τα μοντέλα και για την επιλογή του k.

Το έβγαλα σε ξεχωριστό module γιατί αλλιώς η εξάρτηση γινόταν κυκλική: η
επιλογή του k πρέπει να δει ακριβώς τα δεδομένα που θα δει ο K-Means
(εξωτερικοί hosts χωρίς τις ανωμαλίες), αλλά αυτά παράγονταν μέσα στη
μακροσκοπική φάση, δηλαδή μετά. Τώρα ο διαχωρισμός ορίζεται μία φορά και τον
καλούν και τα δύο.
"""
from sklearn.ensemble import IsolationForest

from src.config import CONTAMINATION_RATE, RANDOM_STATE
from src.utils.anonymize import get_internal_hashes
from src.utils.scaling import scale_features


def split_internal_external(df):
    """(εσωτερικοί, εξωτερικοί) hosts με βάση τα hashes του εσωτερικού subnet."""
    internal_hashes = get_internal_hashes()
    mask = df['Host'].isin(internal_hashes)
    return df[mask].copy(), df[~mask].copy()


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


def macro_dataset(df):
    """Τα δεδομένα που βλέπει ο K-Means της μακροσκοπικής ανάλυσης."""
    _, external = split_internal_external(df)
    if external.empty:
        return external, external
    return remove_anomalies(external)


def micro_dataset(df):
    """Τα δεδομένα που βλέπει ο K-Means της μικροσκοπικής ανάλυσης."""
    internal, _ = split_internal_external(df)
    return internal
