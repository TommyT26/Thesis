# src/utils/summary.py
import pandas as pd
from src.config import FEATURE_COLUMNS


def save_cluster_summary(df_clean, cluster_col, output_path, df_anomalies=None):
    """
    Δημιουργεί ένα ευανάγνωστο CSV με τα προφίλ των clusters:
    πλήθος hosts, ποσοστό (%) και μέσες/διάμεσες τιμές των χαρακτηριστικών.
    """
    df_all = df_clean.copy()
    # Το -1 σημαίνει noise στον DBSCAN, αλλά "δεν έγινε clustering" στο K-Means (π.χ. πολύ λίγα hosts)
    minus_one_label = 'DBSCAN_Noise (-1)' if cluster_col == 'DBSCAN_Cluster' else 'Unclustered (-1)'
    df_all['Profile_Group'] = df_all[cluster_col].apply(
        lambda c: minus_one_label if c == -1 else f'Cluster {c}'
    )

    # Αν υπάρχουν ανωμαλίες (από το Isolation Forest), τις προσθέτουμε ως ξεχωριστή γραμμή σύγκρισης
    if df_anomalies is not None and not df_anomalies.empty:
        anom = df_anomalies.copy()
        anom['Profile_Group'] = 'Anomalies (IsoForest)'
        df_all = pd.concat([df_all, anom], ignore_index=True)

    total_hosts = len(df_all)
    # Εκτός από τα χαρακτηριστικά του μοντέλου, στις συνόψεις μπαίνουν και όσα
    # κρατήθηκαν μόνο για την ερμηνεία (βγήκαν από το μοντέλο ως σχεδόν
    # διπλότυπα, αλλά εξηγούν τι κάνει η κάθε ομάδα).
    interpretive = ['Unique_Ports', 'Flows_per_Peer', 'UDP_Ratio']
    metrics = ['Total_Bytes'] + FEATURE_COLUMNS + interpretive
    # Κρατάμε τις στήλες χωρίς διπλότυπα με τη σειρά που ορίστηκαν
    metrics = list(dict.fromkeys(m for m in metrics if m in df_all.columns))

    # Υπολογισμός Πλήθους, Ποσοστού, Μέσου Όρου και Διαμέσου Bytes
    counts = df_all.groupby('Profile_Group').size().rename('Hosts_Count')
    pct = ((counts / total_hosts) * 100).round(2).rename('Percentage_%')
    means = df_all.groupby('Profile_Group')[metrics].mean().round(2)
    median_bytes = df_all.groupby('Profile_Group')['Total_Bytes'].median().round(2).rename('Median_Bytes')

    summary = pd.concat([counts, pct, median_bytes, means], axis=1).reset_index()
    summary = summary.sort_values(by='Hosts_Count', ascending=False)

    summary.to_csv(output_path, index=False)
    print(f"    - [SUMMARY] Αποθηκεύτηκε σύνοψη προφίλ στο: {output_path}")