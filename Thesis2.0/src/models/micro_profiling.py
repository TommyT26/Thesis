import os
import glob
import pandas as pd
from sklearn.cluster import KMeans

from src.config import PATHS, RANDOM_STATE
from src.utils.scaling import scale_features
from src.models.data_split import micro_dataset
from src.models.dbscan_clustering import fit_dbscan
from src.models.select_k import choose_k
from src.utils.summary import save_cluster_summary


def run(n_clusters=None):
    """
    Μικροσκοπική ανάλυση.

    n_clusters: αν δοθεί, χρησιμοποιείται ως έχει (π.χ. από --k-micro).
    Αλλιώς το ζητάω από τον χρήστη δείχνοντας τους δείκτες εγκυρότητας. Σε μη
    διαδραστική εκτέλεση πέφτει στο MICRO_CLUSTERS του config.py.
    """
    print("\n--- Έναρξη: Μικροσκοπική Ανάλυση (Εσωτερικό Δίκτυο) ---")

    feature_files = glob.glob(os.path.join(PATHS['features'], "features_*.csv"))

    if n_clusters is None:
        n_clusters = choose_k('micro')
    print(f" -> Αριθμός clusters K-Means: {n_clusters}")

    for file_path in feature_files:
        filename = os.path.basename(file_path)
        print(f" -> Επεξεργασία: {filename}")

        df = pd.read_csv(file_path)
        df_internal = micro_dataset(df)
        out_file = filename.replace("features_", "micro_")

        if len(df_internal) < n_clusters:
            print(f"    - Προειδοποίηση: Βρέθηκαν μόνο {len(df_internal)} εσωτερικές IP. Παράλειψη K-Means.")
            # Η στήλη Cluster γράφεται ακόμη κι όταν δεν έγινε ομαδοποίηση (-1
            # = χωρίς ομάδα), όπως κάνει και η μακροσκοπική φάση. Χωρίς αυτήν,
            # τα επόμενα στάδια που διαβάζουν usecols=['Host','Cluster'] (ωριαίο
            # προφίλ, γραφήματα) σκάνε με σφάλμα στήλης αντί να παραλείψουν.
            df_internal['Cluster'] = -1
            df_internal.to_csv(os.path.join(PATHS['results_micro'], out_file), index=False)
            continue

        # K-Means πάνω σε log1p + standardized χαρακτηριστικά
        X = scale_features(df_internal)
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        df_internal['Cluster'] = kmeans.fit_predict(X)

        # DBSCAN (label -1 = noise)
        df_internal['DBSCAN_Cluster'], _ = fit_dbscan(X, name="micro")

        # Αποθήκευση
        df_internal.to_csv(os.path.join(PATHS['results_micro'], out_file), index=False)
        sum_kmeans = filename.replace("features_", "summary_kmeans_micro_")
        sum_dbscan = filename.replace("features_", "summary_dbscan_micro_")
        save_cluster_summary(df_internal, 'Cluster', os.path.join(PATHS['results_micro'], sum_kmeans))
        save_cluster_summary(df_internal, 'DBSCAN_Cluster', os.path.join(PATHS['results_micro'], sum_dbscan))

        print(f"    - Βρέθηκαν και αναλύθηκαν {len(df_internal)} εσωτερικές IP.")

    print("--- Ολοκληρώθηκε ---")


if __name__ == "__main__":
    run()
