import pandas as pd
import numpy as np
import os
import hashlib
import hmac
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from dotenv import load_dotenv

# *---- 1. ΡΥΘΜΙΣΕΙΣ & ΦΟΡΤΩΣΗ ΚΛΕΙΔΙΟΥ ----
INPUT_FILE = 'final_profiles/host_behavioral_vectors.csv'
OUTPUT_FOLDER = 'clustering_results_micro/'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

load_dotenv()
SALT_KEY_STR = os.getenv("THESIS_SALT_KEY")
if not SALT_KEY_STR:
    raise ValueError("ΣΦΑΛΜΑ: Δεν βρέθηκε το κλειδί στο .env")
SALT_KEY_BYTES = SALT_KEY_STR.encode('utf-8')

print("--- ΕΝΑΡΞΗ ΜΙΚΡΟΣΚΟΠΙΚΗΣ ΑΝΑΛΥΣΗΣ (Εσωτερικοί Hosts) ---")

# *---- 2. ΔΗΜΙΟΥΡΓΙΑ HASHES ΓΙΑ ΤΟ ΤΟΠΙΚΟ ΔΙΚΤΥΟ ----
# Υποθέτουμε ότι το δίκτυο είναι το 192.168.7.x
SUBNET = "192.168.7."
internal_hashes = set()

for i in range(1, 255):
    ip_str = f"{SUBNET}{i}"
    msg_bytes = ip_str.encode('utf-8')
    hash_val = hmac.new(SALT_KEY_BYTES, msg_bytes, hashlib.sha256).hexdigest()[:16]
    internal_hashes.add(hash_val)

print(f"Δημιουργήθηκαν {len(internal_hashes)} hashes για το υποδίκτυο {SUBNET}x")

# *---- 3. ΦΟΡΤΩΣΗ ΚΑΙ ΦΙΛΤΡΑΡΙΣΜΑ ΔΕΔΟΜΕΝΩΝ ----
df_all = pd.read_csv(INPUT_FILE)

features_to_use = [
    "Log_Total_Bytes", "Log_Total_Flows", "Log_Unique_Peers", 
    "Out_In_Ratio", "Asymmetry", "Flows_per_Peer", 
    "TCP_Ratio", "UDP_Ratio", "Port_Entropy"
]
df_all = df_all.dropna(subset=features_to_use).copy()

# Κρατάμε ΜΟΝΟ τους εσωτερικούς Hosts
df = df_all[df_all['Host'].isin(internal_hashes)].copy()

print(f"Σύνολο ΕΣΩΤΕΡΙΚΩΝ Hosts προς ανάλυση: {len(df)}")

if len(df) == 0:
    print("Δεν βρέθηκαν εσωτερικοί hosts! Έλεγξε το SUBNET (π.χ. μήπως είναι 192.168.1. ή 10.0.0.).")
    exit()

# *---- 4. SCALING (ΚΑΝΟΝΙΚΟΠΟΙΗΣΗ) ----
scaler = RobustScaler()
X_scaled = scaler.fit_transform(df[features_to_use])

# *---- 5. ΕΦΑΡΜΟΓΗ K-MEANS ----
# Επειδή οι εσωτερικοί χρήστες είναι λίγοι, 3 Clusters είναι ιδανικά
NUM_CLUSTERS = min(3, len(df)) 
print(f"\nΕκτέλεση K-Means με {NUM_CLUSTERS} ομάδες...")

kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
df['Behavior_Group'] = kmeans.fit_predict(X_scaled)
df['Behavior_Group'] = "Internal_Role_" + df['Behavior_Group'].astype(str)

# *---- 6. ΟΠΤΙΚΟΠΟΙΗΣΗ (PCA) ----
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PCA_1'] = X_pca[:, 0]
df['PCA_2'] = X_pca[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PCA_1', y='PCA_2', 
    hue='Behavior_Group', 
    palette="Set1",
    data=df, 
    alpha=0.8, s=80
)
for line in range(0,df.shape[0]):
    plt.text(df.PCA_1.iloc[line]+0.2, df.PCA_2.iloc[line], 
             df.Host.iloc[line][:4], horizontalalignment='left', size='small', color='black')

plt.title('Οπτικοποίηση Εσωτερικών Hosts (Micro-Analysis)')
plt.xlabel('Κύρια Συνιστώσα 1')
plt.ylabel('Κύρια Συνιστώσα 2')
plt.legend(title='Ρόλος')
plt.grid(True, linestyle='--', alpha=0.5)

plot_path = os.path.join(OUTPUT_FOLDER, 'internal_clusters_pca.png')
plt.savefig(plot_path, dpi=300)

# *---- 7. ΕΞΑΓΩΓΗ ΠΡΟΦΙΛ ----
profile_cols = features_to_use + ['Total_Bytes']
cluster_profiles = df.groupby('Behavior_Group')[profile_cols].median()
cluster_profiles['Host_Count'] = df.groupby('Behavior_Group')['Host'].count()

df.to_csv(os.path.join(OUTPUT_FOLDER, 'internal_hosts_analysis.csv'), index=False)
cluster_profiles.to_csv(os.path.join(OUTPUT_FOLDER, 'internal_profiles.csv'))

print("\n--- Τελικά Προφίλ Εσωτερικών Χρηστών (Medians) ---")
print(cluster_profiles[['Host_Count', 'Total_Bytes', 'Out_In_Ratio', 'TCP_Ratio', 'UDP_Ratio']])