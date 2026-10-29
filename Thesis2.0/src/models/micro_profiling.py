import os
import glob
import pandas as pd
from sklearn.cluster import KMeans

from src.config import PATHS, RANDOM_STATE
from src.utils.scaling import scale_features
from src.models.populations import populations
from src.models.dbscan_clustering import fit_dbscan
from src.models.select_k import choose_k
from src.utils.summary import save_cluster_summary


def analyse_population(key, title, hashes, feature_files, n_clusters=None):
    """
    Ομαδοποίηση ενός πληθυσμού hosts.

    Τα ονόματα των αρχείων βγαίνουν από το κλειδί: για key='micro' δίνουν
    micro_global.csv και summary_kmeans_micro_global.csv, δηλαδή ακριβώς ό,τι
    έγραφε το στάδιο πριν γίνει παραμετρικό. Αυτό δεν είναι λεπτομέρεια —
    αυτά τα ονόματα τα διαβάζουν τα στάδια hourly, stability και visualize.
    """
    print(f"\n -> Πληθυσμός: {title}")
    if n_clusters is None:
        n_clusters = choose_k(key)
    print(f"    Αριθμός clusters K-Means: {n_clusters}")

    for file_path in feature_files:
        filename = os.path.basename(file_path)
        print(f"    - Επεξεργασία: {filename}")

        df = pd.read_csv(file_path)
        df_population = df[df['Host'].isin(hashes)].copy()
        out_file = filename.replace("features_", f"{key}_")

        if len(df_population) < n_clusters:
            print(f"      Προειδοποίηση: Βρέθηκαν μόνο {len(df_population)} IP. "
                  f"Παράλειψη K-Means.")
            # Η στήλη Cluster γράφεται ακόμη κι όταν δεν έγινε ομαδοποίηση (-1
            # = χωρίς ομάδα), όπως κάνει και η μακροσκοπική φάση. Χωρίς αυτήν,
            # τα επόμενα στάδια που διαβάζουν usecols=['Host','Cluster'] (ωριαίο
            # προφίλ, γραφήματα) σκάνε με σφάλμα στήλης αντί να παραλείψουν.
            df_population['Cluster'] = -1
            df_population.to_csv(os.path.join(PATHS['results_micro'], out_file), index=False)
            continue

        # K-Means πάνω σε log1p + standardized χαρακτηριστικά
        X = scale_features(df_population)
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        df_population['Cluster'] = kmeans.fit_predict(X)

        # DBSCAN (label -1 = noise)
        df_population['DBSCAN_Cluster'], _ = fit_dbscan(X, name=key)

        # Αποθήκευση
        df_population.to_csv(os.path.join(PATHS['results_micro'], out_file), index=False)
        sum_kmeans = filename.replace("features_", f"summary_kmeans_{key}_")
        sum_dbscan = filename.replace("features_", f"summary_dbscan_{key}_")
        save_cluster_summary(df_population, 'Cluster',
                             os.path.join(PATHS['results_micro'], sum_kmeans))
        save_cluster_summary(df_population, 'DBSCAN_Cluster',
                             os.path.join(PATHS['results_micro'], sum_dbscan))

        # Προειδοποίηση για κατακερματισμό. Σε πληθυσμούς δεκάδων hosts με
        # λίγους ακραίους, οι δείκτες εγκυρότητας ανεβαίνουν όσο ο K-Means
        # απομονώνει μεμονωμένα σημεία: κάθε μονομελής ομάδα είναι τέλεια
        # συμπαγής και μακριά από τις άλλες. Έτσι ένα υψηλό silhouette μπορεί
        # να σημαίνει «μία πραγματική ομάδα συν μοναχικοί hosts» και όχι δομή.
        # Στο macro δεν συμβαίνει, γιατί 121.000 hosts δεν απομονώνονται.
        sizes = df_population['Cluster'].value_counts()
        tiny = int((sizes < 3).sum())
        print(f"      Βρέθηκαν και αναλύθηκαν {len(df_population)} IP σε "
              f"{len(sizes)} ομάδες, μεγέθη {sorted(sizes.tolist(), reverse=True)}.")
        if tiny and tiny * 2 >= len(sizes):
            print(f"      Προσοχή: {tiny} από τις {len(sizes)} ομάδες έχουν "
                  f"λιγότερα από 3 μέλη. Οι δείκτες ανεβαίνουν όταν "
                  f"απομονώνονται μεμονωμένοι hosts, οπότε η τιμή τους εδώ δεν "
                  f"σημαίνει ισχυρή δομή αλλά μία ομάδα συν ακραίους.")


def run(n_clusters=None):
    """
    Μικροσκοπική ανάλυση, για κάθε πληθυσμό hosts χωριστά.

    Πληθυσμοί είναι το εργαστήριο και κάθε υποδίκτυο του SUBNET_PREFIXES.
    Παίρνουν όλοι την ίδια μεταχείριση — ίδια χαρακτηριστικά, ίδια
    κλιμάκωση, ίδιοι δείκτες για το k — ώστε η σύγκριση μεταξύ τους να
    σημαίνει κάτι.

    n_clusters: αν δοθεί (π.χ. από --k-micro), ισχύει για ΟΛΟΥΣ τους
    πληθυσμούς. Αλλιώς κάθε πληθυσμός παίρνει το δικό του k από τους δείκτες
    εγκυρότητας του σταδίου kselect, ή από το config σε μη διαδραστική
    εκτέλεση.
    """
    print("\n--- Έναρξη: Μικροσκοπική Ανάλυση (ανά πληθυσμό) ---")

    feature_files = glob.glob(os.path.join(PATHS['features'], "features_*.csv"))
    if not feature_files:
        print(f"[!] Δεν βρέθηκαν αρχεία στον φάκελο {PATHS['features']}")
        return

    for key, title, hashes in populations():
        analyse_population(key, title, hashes, feature_files, n_clusters)

    print("--- Ολοκληρώθηκε ---")


if __name__ == "__main__":
    run()
