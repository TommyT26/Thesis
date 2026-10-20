import os
import glob
import numpy as np
import pandas as pd
from src.config import (PATHS, BUSINESS_HOURS, PARTIAL_DAYS,
                        WELL_KNOWN_PORT_MAX, P2P_PORTS)

HOURS_PER_DAY = 24

TEMPORAL_COLUMNS = ['Active_Hours', 'Active_Days', 'Hour_Entropy',
                    'Business_Ratio', 'Weekend_Ratio']

PORT_COLUMNS = ['Unique_Ports', 'Port_Entropy', 'Well_Known_Ratio', 'P2P_Port_Ratio']


def port_features(port_counts):
    """
    Χαρακτηριστικά θυρών ανά host, από τα πλήθη ροών ανά (Host, Service_Port).

      Unique_Ports     πόσες διαφορετικές θύρες υπηρεσίας
      Port_Entropy     εντροπία Shannon της κατανομής ροών στις θύρες (nats).
                       0 = όλη η κίνηση σε μία θύρα. Μεγαλώνει όσο η κίνηση
                       απλώνεται σε περισσότερες θύρες πιο ομοιόμορφα.
      Well_Known_Ratio ποσοστό ροών σε θύρες < 1024 (καταχωρημένες υπηρεσίες)
      P2P_Port_Ratio   ποσοστό ροών σε τεκμηριωμένες θύρες P2P

    Προσοχή στην ερμηνεία: η εντροπία από μόνη της ΔΕΝ ξεχωρίζει σάρωση από
    BitTorrent — και οι δύο τη δίνουν υψηλή, και το μέτρησα: τα
    ενδοτεταρτημοριακά εύρη των δύο ομάδων επικαλύπτονται. Ούτε τα
    Well_Known_Ratio και P2P_Port_Ratio αρκούν. Αυτό που τελικά τις χωρίζει
    είναι η ΕΠΑΝΕΜΦΑΝΙΣΗ θυρών από μέρα σε μέρα (peer_recurrence.py). Η
    εντροπία εδώ κάνει κάτι πιο περιορισμένο αλλά σαφές: αποκλείει τους hosts
    τύπου υπηρεσίας, που συγκεντρώνονται σε ελάχιστες θύρες.
    """
    pc = pd.concat(port_counts, ignore_index=True).groupby(
        ['Host', 'Service_Port'], as_index=False)['Flows'].sum()

    total = pc.groupby('Host')['Flows'].sum()
    share = pc['Flows'] / pc['Host'].map(total)
    pc['plogp'] = -share * np.log(share.where(share > 0, 1.0))

    # Γνωστός περιορισμός: το Service_Port == 0 (ICMP/IGMP, πρωτόκολλα χωρίς
    # θύρες) περνά εδώ ως «καταχωρημένη θύρα», αφού 0 < 1024. Είναι 0,27% των
    # γραμμών, και από τους 79 hosts της ομάδας αναφοράς μόνο δύο το αγγίζουν, 
    # κανένας δεν το έχει κυρίαρχο. Δεν αλλάζω τον ορισμό ενός
    # χαρακτηριστικού που έχει ήδη παραγάγει ελεγμένα αποτελέσματα για 0,27%·
    # το καταγράφω. Ο έλεγχος στο port_semantics.py το εξαιρεί.
    well_known = pc[pc['Service_Port'] < WELL_KNOWN_PORT_MAX].groupby('Host')['Flows'].sum()
    p2p = pc[pc['Service_Port'].isin(P2P_PORTS)].groupby('Host')['Flows'].sum()

    return pd.DataFrame({
        'Unique_Ports': pc.groupby('Host').size(),
        'Port_Entropy': pc.groupby('Host')['plogp'].sum(),
        'Well_Known_Ratio': (well_known / total).fillna(0.0),
        'P2P_Port_Ratio': (p2p / total).fillna(0.0),
    }).fillna(0.0)


def host_view(df, host, peer, out_dir, in_dir):
    """Η όψη ενός από τα δύο άκρα του bidirectional flow. out_dir/in_dir: 'AtoB' ή 'BtoA'."""
    return pd.DataFrame({
        'Host': df[host],
        'Peer': df[peer],
        'Service_Port': df['Service_Port'],
        'Protocol': df['Protocol'],
        'Bytes_Out': df[f'Bytes_{out_dir}'],
        'Bytes_In': df[f'Bytes_{in_dir}'],
        'Flows': df['Flows_AtoB'] + df['Flows_BtoA'],
    })


def temporal_features(files):
    """
    Χρονικά χαρακτηριστικά ανά host από τους πίνακες temporal_*.csv.

      Active_Hours   πόσες διαφορετικές ώρες του 24ώρου είναι ενεργός ο host
      Active_Days    σε πόσες διαφορετικές ημέρες εμφανίζεται
      Hour_Entropy   εντροπία Shannon της κατανομής ανά ώρα, κανονικοποιημένη
                     στο [0, 1]: 0 = όλη η κίνηση σε μία ώρα (ανθρώπινο ή
                     περιοδικό μοτίβο), 1 = ομοιόμορφα στο 24ωρο (server/IoT)
      Business_Ratio ποσοστό ροών εντός ωραρίου
      Weekend_Ratio  ποσοστό ροών Σάββατο/Κυριακή

    Επιστρέφει DataFrame με index το Host, ή None αν δεν υπάρχουν αρχεία.
    """
    if not files:
        return None

    print(f" -> Χρονικά χαρακτηριστικά από {len(files)} αρχεία...")
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    if PARTIAL_DAYS:
        before = len(t)
        t = t[~t['Date'].astype('int64').isin(PARTIAL_DAYS)]
        print(f"    - Εξαιρέθηκαν ημέρες με ελλιπή κάλυψη: {before - len(t)} γραμμές "
              f"({', '.join(str(d) for d in PARTIAL_DAYS)})")
    if t.empty:
        return None

    t['Hour'] = t['Hour'].astype('int16')
    t['Date'] = t['Date'].astype('int64')
    t['DayOfWeek'] = t['DayOfWeek'].astype('int16')

    total = t.groupby('Host')['Flows'].sum()

    # Κατανομή ανά ώρα του 24ώρου (αθροισμένη σε όλες τις ημέρες) -> εντροπία
    per_hour = t.groupby(['Host', 'Hour'])['Flows'].sum().reset_index()
    share = per_hour['Flows'] / per_hour['Host'].map(total)
    per_hour['plogp'] = -share * np.log(share.where(share > 0, 1.0))
    entropy = per_hour.groupby('Host')['plogp'].sum() / np.log(HOURS_PER_DAY)

    start, end = BUSINESS_HOURS
    business = t[t['Hour'].between(start, end - 1)].groupby('Host')['Flows'].sum()
    weekend = t[t['DayOfWeek'] >= 5].groupby('Host')['Flows'].sum()

    features = pd.DataFrame({
        'Active_Hours': t.groupby('Host')['Hour'].nunique(),
        'Active_Days': t.groupby('Host')['Date'].nunique(),
        'Hour_Entropy': entropy,
        'Business_Ratio': (business / total).fillna(0.0),
        'Weekend_Ratio': (weekend / total).fillna(0.0),
    })
    return features.fillna(0.0)


def temporal_path_for(bidir_path):
    """Το temporal_*.csv που αντιστοιχεί σε ένα bidir_*.csv."""
    return os.path.join(os.path.dirname(bidir_path),
                        os.path.basename(bidir_path).replace("bidir_", "temporal_", 1))


def build_profiles(bidir_files, temporal_files=None):
    """
    Χτίζει τα προφίλ των hosts από μια λίστα bidir αρχείων.

    Το έβγαλα σε χωριστή συνάρτηση ώστε να το καλώ και για μία ημέρα (ανάλυση
    ευστάθειας) και για όλες μαζί (Φάση 4), χωρίς διπλό κώδικα.
    """
    if temporal_files is None:
        temporal_files = [p for p in (temporal_path_for(f) for f in bidir_files)
                          if os.path.exists(p)]

    # Κρατάμε μόνο μικρά ενδιάμεσα αποτελέσματα ανά ημέρα
    daily_sums = []   # αθροίσματα ανά host
    peer_pairs = []   # μοναδικά ζεύγη (Host, Peer)
    port_counts = []  # πλήθη ροών ανά (Host, Service_Port)

    for file_path in bidir_files:
        filename = os.path.basename(file_path)
        print(f"      - Εξαγωγή μετρικών από: {filename}...")

        df = pd.read_csv(file_path)

        df_day = pd.concat([
            host_view(df, 'IP_A', 'IP_B', out_dir='AtoB', in_dir='BtoA'),
            host_view(df, 'IP_B', 'IP_A', out_dir='BtoA', in_dir='AtoB'),
        ], ignore_index=True)

        df_day['TCP_Flows'] = df_day['Flows'].where(df_day['Protocol'] == 'TCP', 0)
        df_day['UDP_Flows'] = df_day['Flows'].where(df_day['Protocol'] == 'UDP', 0)

        daily_sums.append(df_day.groupby('Host')[
            ['Bytes_In', 'Bytes_Out', 'Flows', 'TCP_Flows', 'UDP_Flows']
        ].sum().reset_index())

        peer_pairs.append(df_day[['Host', 'Peer']].drop_duplicates())
        port_counts.append(df_day.groupby(
            ['Host', 'Service_Port'], as_index=False)['Flows'].sum())

    # Τα αθροίσματα προστίθενται σωστά μεταξύ ημερών
    sums = pd.concat(daily_sums, ignore_index=True).groupby('Host').sum()

    # Τα μοναδικά peers/ports δεν προστίθενται: ένας peer που εμφανίζεται 8 μέρες
    # θα μετριόταν 8 φορές. Πετάω πρώτα τα διπλότυπα ζεύγη και μετά μετράω.
    unique_peers = pd.concat(peer_pairs, ignore_index=True).drop_duplicates() \
        .groupby('Host').size().rename('Unique_Peers')
    # Τα πλήθη ανά θύρα δίνουν και το Unique_Ports (πλήθος γραμμών ανά host),
    # οπότε δεν χρειάζεται ξεχωριστός συσσωρευτής για τα μοναδικά ζεύγη.
    p = sums.join(unique_peers).join(port_features(port_counts))

    # Παράγωγα χαρακτηριστικά
    p['Total_Bytes'] = p['Bytes_In'] + p['Bytes_Out']
    p['Total_Flows'] = p['Flows']
    p['Out_Ratio'] = (p['Bytes_Out'] / p['Total_Bytes']).where(p['Total_Bytes'] > 0, 0.5)  # 1 = μόνο αποστολέας
    p['TCP_Ratio'] = p['TCP_Flows'] / p['Total_Flows']
    p['UDP_Ratio'] = p['UDP_Flows'] / p['Total_Flows']
    p['Avg_Bytes_Per_Flow'] = p['Total_Bytes'] / p['Total_Flows']
    # Ροές ανά peer: ο σαρωτής αγγίζει πολλούς peers με ελάχιστες ροές ο καθένας,
    # ενώ ένας peer σε swarm ανταλλάσσει επανειλημμένα με τους ίδιους.
    p['Flows_per_Peer'] = p['Total_Flows'] / p['Unique_Peers'].clip(lower=1)

    # Χρονικά χαρακτηριστικά
    temporal = temporal_features(temporal_files)
    if temporal is None:
        print("      ! Δεν βρέθηκαν αρχεία temporal_*.csv. Τα χρονικά χαρακτηριστικά\n"
              "        μηδενίζονται. Ξανατρέξε:  python main.py --step aggregate")
        for col in TEMPORAL_COLUMNS:
            p[col] = 0.0
    else:
        p = p.join(temporal)
        missing = int(p['Active_Days'].isna().sum())
        if missing:
            print(f"      ! {missing} hosts χωρίς χρονικά δεδομένα (μηδενίζονται).")
        for col in TEMPORAL_COLUMNS:
            p[col] = p[col].fillna(0.0)

    return p[[
        'Bytes_In', 'Bytes_Out', 'Total_Bytes', 'Total_Flows',
        'Unique_Peers', 'Out_Ratio', 'TCP_Ratio', 'UDP_Ratio',
        'Avg_Bytes_Per_Flow', 'Flows_per_Peer'
    ] + PORT_COLUMNS + TEMPORAL_COLUMNS]


def run():
    print("\n--- Έναρξη: Εξαγωγή Χαρακτηριστικών (Map-Reduce Profiling) ---")

    bidir_files = glob.glob(os.path.join(PATHS['bidirectional_data'], "bidir_*.csv"))
    if not bidir_files:
        print(f"Προειδοποίηση: Δεν βρέθηκαν αρχεία στον φάκελο {PATHS['bidirectional_data']}")
        return

    print(f" -> Εντοπίστηκαν {len(bidir_files)} αρχεία προς συνένωση.")
    final_profiles = build_profiles(bidir_files).reset_index()

    output_path = os.path.join(PATHS['features'], "features_global.csv")
    final_profiles.to_csv(output_path, index=False)
    print(f" -> Το συνολικό αρχείο αποθηκεύτηκε στο: {output_path} ({len(final_profiles)} hosts)")

    print("--- Ολοκληρώθηκε ---")


if __name__ == "__main__":
    run()
