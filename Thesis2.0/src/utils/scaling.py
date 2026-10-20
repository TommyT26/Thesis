import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import LOG_FEATURES, LINEAR_FEATURES, RATIO_FEATURES


def scale_features(df, log_columns=LOG_FEATURES, linear_columns=LINEAR_FEATURES,
                   ratio_columns=RATIO_FEATURES):
    """
    log1p στα μεγέθη με βαριά ουρά, StandardScaler σε όλα.

    Τα bytes και οι ροές είναι τάξεις μεγέθους μεγαλύτερα από τα Unique_Ports,
    οπότε χωρίς αυτό το βήμα οι αποστάσεις του K-Means και του DBSCAN
    καθορίζονται σχεδόν μόνο από τα bytes. Τα μικρά φραγμένα πλήθη
    (Active_Hours, Active_Days) και οι λόγοι στο [0, 1] δεν παίρνουν log.
    Γυρίζει numpy array· το df δεν αλλάζει.
    """
    names = list(log_columns) + list(linear_columns) + list(ratio_columns)
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise KeyError(
            f"Λείπουν χαρακτηριστικά από τα προφίλ: {', '.join(missing)}. "
            "Αν πρόκειται για τα χρονικά χαρακτηριστικά, ξανατρέξε: "
            "python main.py --step aggregate features")

    X_log = np.log1p(df[log_columns].fillna(0).clip(lower=0)).to_numpy()
    X_linear = df[linear_columns].fillna(0).to_numpy()
    X_ratio = df[ratio_columns].fillna(0).to_numpy()
    X = np.hstack([X_log, X_linear, X_ratio]).astype(float)

    # Νεκρές διαστάσεις: μια στήλη με μηδενική διασπορά δεν ξεχωρίζει κανένα
    # host, αλλά μετράει κανονικά στις αποστάσεις. Έτσι βρήκα το Total_Packets,
    # που ήταν πάντα 0 επειδή δεν υπάρχει καν στήλη packets στο export.
    constant = [name for name, col in zip(names, X.T) if np.ptp(col) == 0]
    if constant:
        print(f"    - Προειδοποίηση: σταθερά χαρακτηριστικά χωρίς πληροφορία: "
              f"{', '.join(constant)} (δες τις λίστες χαρακτηριστικών στο config.py)")

    return StandardScaler().fit_transform(X)
