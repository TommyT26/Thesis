import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# *---- 1. ΡΥΘΜΙΣΕΙΣ ----
INPUT_FILE = 'final_profiles/host_behavioral_vectors.csv'
OUTPUT_FOLDER = 'clustering_results_v2/'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print("--- ΕΝΑΡΞΗ ADVANCED PIPELINE (V2: Isolation Forest + K-Means) ---")

# *---- 2. ΦΟΡΤΩΣΗ ΚΑΙ ΕΠΙΛΟΓΗ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ ----
print(f"Φόρτωση δεδομένων από: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

# Επιλέγουμε τα χαρακτηριστικά (όπως και στην V1)
features_to_use = [
    "Log_Total_Bytes", 
    "Log_Total_Flows", 
    "Log_Unique_Peers", 
    "Out_In_Ratio", 
    "Asymmetry", 
    "Flows_per_Peer", 
    "TCP_Ratio", 
    "UDP_Ratio", 
    "Port_Entropy"
]

df = df.dropna(subset=features_to_use).copy()
X = df[features_to_use]
print(f"Σύνολο Hosts προς ανάλυση: {len(df):,}")

# *---- 3. SCALING (ΚΑΝΟΝΙΚΟΠΟΙΗΣΗ) ----
print("\n[Βήμα 1] Κανονικοποίηση με RobustScaler...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# *---- 4. ΕΝΤΟΠΙΣΜΟΣ ΑΝΩΜΑΛΙΩΝ (ISOLATION FOREST) ----
print("\n[Βήμα 2] Εκτέλεση Isolation Forest για εντοπισμό Scanners/Outliers...")
# Το contamination=0.015 σημαίνει ότι περιμένουμε ~1.5% των hosts να είναι ανωμαλίες
iso_forest = IsolationForest(contamination=0.015, random_state=42, n_jobs=-1)
# Επιστρέφει -1 για ανωμαλία, 1 για φυσιολογικό
df['Is_Anomaly'] = iso_forest.fit_predict(X_scaled)

# Διαχωρισμός των δεδομένων σε Normal και Anomalies
df_normal = df[df['Is_Anomaly'] == 1].copy()
df_anomalies = df[df['Is_Anomaly'] == -1].copy()

print(f"  -> Εντοπίστηκαν {len(df_anomalies)} ανωμαλίες (outliers).")
print(f"  -> Φυσιολογικοί Hosts (προς Clustering): {len(df_normal):,}")

# *---- 5. ΕΦΑΡΜΟΓΗ K-MEANS ΜΟΝΟ ΣΤΟΥΣ ΦΥΣΙΟΛΟΓΙΚΟΥΣ HOSTS ----
print("\n[Βήμα 3] Εκτέλεση K-Means (4 clusters) ΜΟΝΟ στην κανονική κίνηση...")
NUM_CLUSTERS = 4
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)

# Προσοχή: Πρέπει να κάνουμε scale ΞΑΝΑ, μόνο τα normal δεδομένα αυτή τη φορά!
scaler_normal = RobustScaler()
X_normal = df_normal[features_to_use]
X_scaled_normal = scaler_normal.fit_transform(X_normal)

df_normal['Behavior_Group'] = kmeans.fit_predict(X_scaled_normal)
# Μετατρέπουμε τα νούμερα σε String για να είναι ξεκάθαρα
df_normal['Behavior_Group'] = "Normal_Cluster_" + df_normal['Behavior_Group'].astype(str)

# Αντιστοιχίζουμε την ετικέτα "Anomaly" στα outliers
df_anomalies['Behavior_Group'] = "Anomaly"

# *---- 6. ΕΝΟΠΟΙΗΣΗ ΔΕΔΟΜΕΝΩΝ ΚΑΙ PCA ΟΠΤΙΚΟΠΟΙΗΣΗ ----
print("\n[Βήμα 4] Ενοποίηση και Μείωση Διαστάσεων (PCA) για όλο το dataset...")
df_final = pd.concat([df_normal, df_anomalies], ignore_index=True)

# Κάνουμε PCA σε ΟΛΑ τα δεδομένα για το γράφημα (χρησιμοποιώντας το αρχικό X_scaled)
pca = PCA(n_components=2)
# Φτιάχνουμε ξανά το X_scaled με τη νέα σειρά του df_final
X_scaled_final = scaler.fit_transform(df_final[features_to_use])
X_pca = pca.fit_transform(X_scaled_final)

df_final['PCA_1'] = X_pca[:, 0]
df_final['PCA_2'] = X_pca[:, 1]

# ΓΡΑΦΗΜΑ 1: Όλα τα δεδομένα (Anomalies με κόκκινο)
plt.figure(figsize=(12, 8))
# Επιλέγουμε χρώματα: Κόκκινο για τα Anomaly, άλλες αποχρώσεις για τα Normal
unique_groups = sorted(df_final['Behavior_Group'].unique())
palette = {group: "red" if group == "Anomaly" else sns.color_palette("viridis", NUM_CLUSTERS)[i-1] 
           for i, group in enumerate(unique_groups)}

sns.scatterplot(
    x='PCA_1', y='PCA_2', 
    hue='Behavior_Group', 
    palette=palette,
    data=df_final, 
    alpha=0.6,
    s=30 # Μέγεθος κουκκίδων
)
plt.title('Οπτικοποίηση V2: Εντοπισμός Ανωμαλιών & Υπο-ομάδες Φυσιολογικής Κίνησης')
plt.xlabel('Κύρια Συνιστώσα 1 (PCA_1)')
plt.ylabel('Κύρια Συνιστώσα 2 (PCA_2)')
plt.legend(title='Ομάδα Συμπεριφοράς', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout() # Για να μη κοπεί το legend
plt.grid(True, linestyle='--', alpha=0.5)

plot_path = os.path.join(OUTPUT_FOLDER, 'advanced_clusters_pca.png')
plt.savefig(plot_path, dpi=300)
print(f"Αποθηκεύτηκε το γράφημα: {plot_path}")

# *---- 7. ΕΞΑΓΩΓΗ ΠΡΟΦΙΛ ----
print("\n[Βήμα 5] Υπολογισμός Στατιστικών Προφίλ...")
profile_cols = features_to_use + ['Total_Bytes']
cluster_profiles = df_final.groupby('Behavior_Group')[profile_cols].median()
cluster_profiles['Host_Count'] = df_final.groupby('Behavior_Group')['Host'].count()

output_csv = os.path.join(OUTPUT_FOLDER, 'hosts_final_analysis.csv')
df_final.to_csv(output_csv, index=False)

profiles_csv = os.path.join(OUTPUT_FOLDER, 'behavioral_profiles.csv')
cluster_profiles.to_csv(profiles_csv)

print(f"Αποθηκεύτηκαν τα τελικά αρχεία στον φάκελο {OUTPUT_FOLDER}")
print("\n--- Τελικά Προφίλ Συμπεριφοράς (Medians) ---")
print(cluster_profiles[['Host_Count', 'Total_Bytes', 'Out_In_Ratio', 'TCP_Ratio', 'UDP_Ratio', 'Port_Entropy']])
print("\nΔιαδικασία V2 Ολοκληρώθηκε Επιτυχώς!")
