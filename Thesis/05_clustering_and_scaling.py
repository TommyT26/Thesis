import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# *---- 1. ΡΥΘΜΙΣΕΙΣ ----
INPUT_FILE = 'final_profiles/host_behavioral_vectors.csv'
OUTPUT_FOLDER = 'clustering_results/'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print("--- ΕΝΑΡΞΗ CLUSTERING & SCALING ---")

# *---- 2. ΦΟΡΤΩΣΗ ΚΑΙ ΕΠΙΛΟΓΗ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ ----
print(f"Φόρτωση δεδομένων από: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

# Επιλέγουμε τα features που έχουν νόημα για τον αλγόριθμο.
# Χρησιμοποιούμε τις Log εκδοχές για τα απόλυτα μεγέθη για να αποφύγουμε τεράστιες αποκλίσεις.
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

# Αφαιρούμε τυχόν γραμμές με NaN (αν προέκυψαν από διαιρέσεις με το 0 σε προηγούμενα στάδια)
df = df.dropna(subset=features_to_use).copy()
X = df[features_to_use]

print(f"Σύνολο Hosts προς ανάλυση: {len(df)}")

# *---- 3. SCALING (ΚΑΝΟΝΙΚΟΠΟΙΗΣΗ) ----
print("Κανονικοποίηση δεδομένων με RobustScaler (ανθεκτικό σε outliers)...")
# Ο RobustScaler χρησιμοποιεί διάμεσο (median) και ενδοτεταρτημοριακό εύρος (IQR),
# οπότε δεν επηρεάζεται από ακραία spikes στο δίκτυο.
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# *---- 4. ΕΦΑΡΜΟΓΗ K-MEANS ----
# Εδώ επιλέγουμε 5 clusters ως αρχικό νούμερο (μπορείς να το αλλάξεις)
NUM_CLUSTERS = 5
print(f"Εκτέλεση K-Means με {NUM_CLUSTERS} clusters...")

kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Προαιρετικά: Υπολογισμός Silhouette Score για αξιολόγηση της ποιότητας των clusters
# (Τιμές κοντά στο 1 σημαίνουν εξαιρετικό διαχωρισμό)
if len(df) > 10:
    # Υπολογισμός σε δείγμα αν τα δεδομένα είναι πάρα πολλά για να γλιτώσουμε χρόνο
    sample_size = min(10000, len(X_scaled)) 
    sil_score = silhouette_score(X_scaled[:sample_size], df['Cluster'][:sample_size])
    print(f"Silhouette Score (Ποιότητα Ομαδοποίησης): {sil_score:.3f}")

# *---- 5. ΜΕΙΩΣΗ ΔΙΑΣΤΑΣΕΩΝ (PCA) ΚΑΙ ΟΠΤΙΚΟΠΟΙΗΣΗ ----
print("Μείωση διαστάσεων με PCA σε 2D για οπτικοποίηση...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PCA_1'] = X_pca[:, 0]
df['PCA_2'] = X_pca[:, 1]

# Δημιουργία γραφήματος Scatter
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PCA_1', y='PCA_2', 
    hue='Cluster', 
    palette='viridis', 
    data=df, 
    alpha=0.6
)
plt.title('Οπτικοποίηση Clusters Συμπεριφοράς Δικτύου (PCA)')
plt.xlabel('Κύρια Συνιστώσα 1 (PCA_1)')
plt.ylabel('Κύρια Συνιστώσα 2 (PCA_2)')
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.5)

plot_path = os.path.join(OUTPUT_FOLDER, 'clusters_pca_2d.png')
plt.savefig(plot_path, dpi=300)
print(f"Αποθηκεύτηκε το γράφημα: {plot_path}")

# *---- 6. ΕΞΑΓΩΓΗ ΠΡΟΦΙΛ (ΜΕΣΟΙ ΟΡΟΙ ΑΝΑ CLUSTER) ----
# Ομαδοποιούμε τα αρχικά δεδομένα (όχι τα scaled) για να είναι ερμηνεύσιμα τα νούμερα
cluster_profiles = df.groupby('Cluster')[features_to_use + ['Total_Bytes']].median()
cluster_profiles['Host_Count'] = df.groupby('Cluster')['Host'].count()

# Αποθήκευση αποτελεσμάτων
output_csv = os.path.join(OUTPUT_FOLDER, 'hosts_with_clusters.csv')
df.to_csv(output_csv, index=False)

profiles_csv = os.path.join(OUTPUT_FOLDER, 'cluster_profiles.csv')
cluster_profiles.to_csv(profiles_csv)

print(f"\nΑποθηκεύτηκαν τα αρχεία:")
print(f" - {output_csv} (Αναλυτικά αποτελέσματα ανά host)")
print(f" - {profiles_csv} (Στατιστικά προφίλ ανά cluster)")

print("\n--- Προφίλ Κάθε Ομάδας (Medians) ---")
print(cluster_profiles[['Host_Count', 'Log_Total_Bytes', 'Out_In_Ratio', 'Port_Entropy', 'TCP_Ratio']])
print("\nΔιαδικασία Ολοκληρώθηκε!")