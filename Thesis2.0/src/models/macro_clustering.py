import os
import glob
import pandas as pd
from sklearn.cluster import KMeans

from src.config import PATHS, RANDOM_STATE
from src.utils.scaling import scale_features
from src.models.data_split import macro_dataset
from src.models.dbscan_clustering import fit_dbscan
from src.models.select_k import choose_k
from src.utils.summary import save_cluster_summary


def run(n_clusters=None):
    """
    Μακροσκοπική ανάλυση.

    n_clusters: αν δοθεί, χρησιμοποιείται ως έχει (π.χ. από --k-macro).
    Αλλιώς το ζητάω από τον χρήστη δείχνοντας τους δείκτες εγκυρότητας. Σε μη
    διαδραστική εκτέλεση πέφτει στο K_PER_LEVEL['macro'] του config.py.
    """
    print("\n--- Έναρξη: Μακροσκοπική Ανάλυση (Εξωτερικό Δίκτυο) ---")
    feature_files = glob.glob(os.path.join(PATHS['features'], "features_*.csv"))

    if n_clusters is None:
        n_clusters = choose_k('macro')
    print(f" -> Αριθμός clusters K-Means: {n_clusters}")

    for file_path in feature_files:
        filename = os.path.basename(file_path)
        print(f" -> Επεξεργασία: {filename}")

        df = pd.read_csv(file_path)

        # 1. Εξωτερικοί hosts, χωρίς τις ανωμαλίες του Isolation Forest
        clean_data, anomalies = macro_dataset(df)
        if clean_data.empty and anomalies.empty:
            continue

        # 2. K-Means στους καθαρούς εξωτερικούς. Ο scaler ξαναπροσαρμόζεται στα
        # καθαρά δεδομένα, ώστε οι outliers να μην επηρεάζουν μέση τιμή / απόκλιση.
        X_clean = scale_features(clean_data)
        clean_data['Cluster'] = -1
        if len(clean_data) >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
            clean_data['Cluster'] = kmeans.fit_predict(X_clean)
        else:
            print(f"    - Προειδοποίηση: Μόνο {len(clean_data)} εξωτερικά hosts. Παράλειψη K-Means.")

        # 3. DBSCAN στα ίδια καθαρά δεδομένα: label -1 = noise, δεύτερη γνώμη για τους ύποπτους
        clean_data['DBSCAN_Cluster'], _ = fit_dbscan(X_clean, name="macro")

        # Αποθήκευση
        anom_file = filename.replace("features_", "anomalies_macro_")
        clean_file = filename.replace("features_", "clean_macro_")
        anomalies.to_csv(os.path.join(PATHS['results_macro'], anom_file), index=False)
        clean_data.to_csv(os.path.join(PATHS['results_macro'], clean_file), index=False)
        # Αποθήκευση ευανάγνωστων πινάκων σύνοψης (Summary CSVs)
        sum_kmeans = filename.replace("features_", "summary_kmeans_macro_")
        sum_dbscan = filename.replace("features_", "summary_dbscan_macro_")
        save_cluster_summary(clean_data, 'Cluster', os.path.join(PATHS['results_macro'], sum_kmeans), anomalies)
        save_cluster_summary(clean_data, 'DBSCAN_Cluster', os.path.join(PATHS['results_macro'], sum_dbscan), anomalies)

        print(f"    - Εξωτερικά hosts: {len(clean_data) + len(anomalies)} | Ανωμαλίες: {len(anomalies)}")

    print("--- Ολοκληρώθηκε ---")


if __name__ == "__main__":
    run()
