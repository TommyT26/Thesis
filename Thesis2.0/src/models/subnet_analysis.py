import os
import glob

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.config import (PATHS, BASE_DIR, RANDOM_STATE, INTERNAL_SUBNET,
                        SUBNET_PREFIXES, SUBNET_K_RANGE, SUBNET_MIN_FLOWS_HOURLY)
from src.models.populations import subnet_populations
from src.utils.anonymize import label_subnets, HOST_OCTETS
from src.utils.scaling import scale_features

# Εικασίες για το τι υπηρεσία είναι κάθε θύρα, από το μητρώο της IANA και τη
# συνήθη χρήση. Μπαίνουν στο CSV ως σχόλιο και ΔΕΝ συμμετέχουν σε κανέναν
# υπολογισμό: τα μετρημένα είναι ο αριθμός της θύρας, η κατεύθυνση και τα
# πλήθη. Όποιος διαφωνεί με την ερμηνεία βλέπει τα νούμερα από κάτω.
PORT_NOTES = {
    22: 'SSH',
    53: 'DNS',
    67: 'DHCP',
    80: 'HTTP',
    123: 'NTP',
    443: 'HTTPS',
    6633: 'OpenFlow (παλαιό, de facto)',
    6653: 'OpenFlow (IANA)',
    6783: 'Splashtop, απομακρυσμένη επιφάνεια εργασίας',
    6881: 'BitTorrent',
    6969: 'BitTorrent tracker',
    8080: 'HTTP εναλλακτική',
    9100: 'pdl-datastream, απευθείας εκτύπωση (JetDirect)',
    9595: 'pds, εντοπισμός πράκτορα LANDesk',
    9596: 'cba8, πράκτορας διαχείρισης LANDesk',
    51413: 'Transmission (BitTorrent)',
}

# Ό,τι δεν ανήκει στα υποδίκτυα της ανάλυσης. Δεν το λέω «Διαδίκτυο»: μαζί με
# τους πραγματικούς hosts του διαδικτύου περιλαμβάνει και μερικές δεκάδες
# διευθύνσεις άλλων ιδιωτικών υποδικτύων, που σηκώνουν περίπου το 0,2% της
# κίνησης. Μικρό νούμερο, αλλά μια ετικέτα που λέει «διαδίκτυο» για διευθύνσεις
# RFC1918 είναι ακριβώς το λάθος που ψάχνει να διορθώσει αυτό το στάδιο.
EXTERNAL = 'Εκτός των τριών'


def output_dir():
    path = os.path.join(BASE_DIR, 'results', 'subnets')
    os.makedirs(path, exist_ok=True)
    return path


def file_label(prefix):
    """'192.168.8.' -> '192_168_8', για ονόματα αρχείων."""
    return prefix.rstrip('.').replace('.', '_')


def group_label(prefix):
    """'192.168.8.' -> '192.168.8.x', η ετικέτα που βλέπει ο αναγνώστης."""
    return f"{prefix}x"


def prefixes():
    """
    Τα υποδίκτυα της ανάλυσης, με το δηλωμένο εσωτερικό πρώτο.

    Το INTERNAL_SUBNET είναι πλέον παράγωγο της SUBNET_PREFIXES (το πρώτο
    στοιχείο της), οπότε ο βρόχος απλώς διατηρεί τη σειρά και αφαιρεί τυχόν
    διπλή εγγραφή. Μένει γραμμένος έτσι και όχι ως list(SUBNET_PREFIXES),
    ώστε η πρόθεση να φαίνεται: η πρώτη θέση ΕΙΝΑΙ η ομάδα αναφοράς με την
    οποία συγκρίνονται τα άλλα υποδίκτυα, δεν είναι τυχαία σειρά.
    """
    found = [INTERNAL_SUBNET]
    for prefix in SUBNET_PREFIXES:
        if prefix not in found:
            found.append(prefix)
    return found


def labelled_features():
    """Τα προφίλ με μια στήλη Υποδίκτυο, μόνο για τους hosts των υποδικτύων."""
    path = os.path.join(PATHS['features'], 'features_global.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Δεν βρέθηκε το {path}. Τρέξε πρώτα:  python main.py --step features")

    frame = pd.read_csv(path)
    known = prefixes()

    # Μέλη του πληθυσμού είναι ΜΟΝΟ οι hosts, οκτέτα 1-254. Το ίδιο εύρος για
    # τα τρία υποδίκτυα, το ίδιο που βλέπουν και τα στάδια kselect, micro και
    # stability μέσω του populations.py. Η εναλλακτική — να μπουν και οι
    # διευθύνσεις δικτύου και εκπομπής — δεν είναι ουδέτερη: η 192.168.8.255
    # έχει Out_Ratio 0, TCP_Ratio 0 και 36 bytes ανά ροή, και θα έμπαινε ως
    # ακραίος «host» σε πληθυσμό 25 μηχανών.
    frame['Υποδίκτυο'] = label_subnets(frame['Host'], known, HOST_OCTETS)
    inside = frame[frame['Υποδίκτυο'].notna()].reset_index(drop=True)

    # Οι διευθύνσεις εκπομπής και δικτύου, αν εμφανίζονται στα δεδομένα,
    # αναφέρονται ρητά. Υπάρχουν — απλώς δεν είναι μηχανές.
    special = label_subnets(frame['Host'], known, (0, 255)).dropna()
    print(f" -> {len(inside)} hosts από τους {len(frame)} ανήκουν στα υποδίκτυα "
          f"της ανάλυσης (οκτέτα 1-254).")
    if len(special):
        counts = ', '.join(f'{name} x{n}' for name, n in special.value_counts().items())
        print(f"    Εκτός, ως διευθύνσεις δικτύου ή εκπομπής: {len(special)} "
              f"({counts}). Δεν είναι μηχανές και δεν ομαδοποιούνται.")
    return frame, inside


def write_inventory(every, inside):
    """Πόσες διευθύνσεις και πόση κίνηση έχει κάθε υποδίκτυο."""
    table = inside.groupby('Υποδίκτυο').agg(
        Διευθύνσεις=('Host', 'nunique'),
        Ροές=('Total_Flows', 'sum'),
        Bytes=('Total_Bytes', 'sum'),
        Διάμ_ροές=('Total_Flows', 'median'),
    )
    # ΠΡΟΣΟΧΗ στο ποσοστό: το άθροισμα των Total_Flows ανά host μετράει δύο
    # φορές κάθε ροή που έχει και τα δύο άκρα της μέσα στο ίδιο υποδίκτυο.
    # Το πραγματικό μερίδιο ροών βγαίνει από τα bidir (write_peers) και είναι
    # μικρότερο. Εδώ το γράφω ως «πλευρές ροών» ώστε να μην παρεξηγηθεί.
    total_sides = every['Total_Flows'].sum()
    table['Ποσοστό_πλευρών_%'] = (100 * table['Ροές'] / total_sides).round(2)
    table['GB'] = (table['Bytes'] / 1e9).round(3)
    table = table.drop(columns=['Bytes']).reset_index()

    path = os.path.join(output_dir(), 'subnet_inventory.csv')
    table.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"    - [ΑΠΟΓΡΑΦΗ] {path}")
    print(table.to_string(index=False))
    return table


def write_profiles(inside):
    """Διάμεσοι των χαρακτηριστικών ανά υποδίκτυο: η συμπεριφορά με μια ματιά."""
    columns = ['Total_Flows', 'Total_Bytes', 'Unique_Peers', 'Flows_per_Peer',
               'Avg_Bytes_Per_Flow', 'TCP_Ratio', 'UDP_Ratio', 'Out_Ratio',
               'Unique_Ports', 'Port_Entropy', 'Well_Known_Ratio', 'P2P_Port_Ratio',
               'Active_Hours', 'Active_Days', 'Hour_Entropy', 'Business_Ratio',
               'Weekend_Ratio']
    columns = [c for c in columns if c in inside.columns]
    table = inside.groupby('Υποδίκτυο')[columns].median().round(4).T
    table.index.name = 'Χαρακτηριστικό'

    path = os.path.join(output_dir(), 'subnet_profiles.csv')
    table.to_csv(path, encoding='utf-8-sig')
    print(f"    - [ΠΡΟΦΙΛ] {path}")
    return table


def bidir_frame():
    """Όλα τα bidir αρχεία μαζί, με ετικέτα υποδικτύου στα δύο άκρα."""
    files = sorted(glob.glob(os.path.join(PATHS['bidirectional_data'], 'bidir_*.csv')))
    if not files:
        return None

    frames = [pd.read_csv(f) for f in files]
    frame = pd.concat(frames, ignore_index=True)
    known = prefixes()
    frame['Α'] = label_subnets(frame['IP_A'], known).fillna(EXTERNAL)
    frame['Β'] = label_subnets(frame['IP_B'], known).fillna(EXTERNAL)
    frame['Ροές'] = frame['Flows_AtoB'].fillna(0) + frame['Flows_BtoA'].fillna(0)
    frame['Μπάιτ'] = frame['Bytes_AtoB'].fillna(0) + frame['Bytes_BtoA'].fillna(0)
    print(f" -> {len(frame)} γραμμές bidir από {len(files)} ημέρες.")
    return frame


def write_peers(bidir):
    """
    Με ποιους μιλά κάθε υποδίκτυο.

    Αυτός είναι ο πίνακας που κρίνει αν ένα ιδιωτικό υποδίκτυο είναι «κίνηση
    διαδικτύου» ή υποδομή του ίδιου δικτύου: αν το συντριπτικό ποσοστό των
    ροών του πηγαίνει προς το εργαστήριο και σχεδόν τίποτα προς τα έξω, δεν
    είναι host του διαδικτύου ό,τι κι αν λέει ο τρέχων διαχωρισμός.
    """
    rows = []
    total = bidir['Ροές'].sum()
    for name in sorted(set(bidir['Α']) | set(bidir['Β'])):
        if name == EXTERNAL:
            continue
        mask = (bidir['Α'] == name) | (bidir['Β'] == name)
        sub = bidir[mask]
        other = np.where(sub['Α'] == name, sub['Β'], sub['Α'])
        own = sub['Ροές'].sum()
        for peer, flows in pd.Series(sub['Ροές'].to_numpy()).groupby(other).sum().items():
            rows.append({
                'Υποδίκτυο': name,
                'Μιλά_με': peer,
                'Ροές': int(flows),
                'Ποσοστό_του_υποδικτύου_%': round(100 * flows / own, 2) if own else np.nan,
                'Ποσοστό_του_συνόλου_%': round(100 * flows / total, 2) if total else np.nan,
            })
    table = pd.DataFrame(rows).sort_values(['Υποδίκτυο', 'Ροές'], ascending=[True, False])

    path = os.path.join(output_dir(), 'subnet_peers.csv')
    table.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"    - [ΕΠΑΦΕΣ] {path}")
    print(table.to_string(index=False))
    return table


def write_ports(bidir, top=10):
    """Οι κορυφαίες Service_Port κάθε υποδικτύου, με το μερίδιό τους στην κίνησή του."""
    rows = []
    for name in sorted(set(bidir['Α']) | set(bidir['Β'])):
        if name == EXTERNAL:
            continue
        sub = bidir[(bidir['Α'] == name) | (bidir['Β'] == name)]
        own = sub['Ροές'].sum()
        grouped = sub.groupby(['Service_Port', 'Protocol'])['Ροές'].sum()
        for (port, protocol), flows in grouped.sort_values(ascending=False).head(top).items():
            rows.append({
                'Υποδίκτυο': name,
                'Service_Port': int(port),
                'Protocol': protocol,
                'Ροές': int(flows),
                'Ποσοστό_του_υποδικτύου_%': round(100 * flows / own, 2) if own else np.nan,
                'Εικασία_υπηρεσίας': PORT_NOTES.get(int(port), ''),
            })
    table = pd.DataFrame(rows)

    path = os.path.join(output_dir(), 'subnet_ports.csv')
    table.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"    - [ΘΥΡΕΣ] {path}")
    return table


def busiest_clean_file():
    """Το μεγαλύτερο clean_*.csv: η ημέρα με την περισσότερη κίνηση."""
    files = glob.glob(os.path.join(PATHS['processed_data'], 'clean_*.csv'))
    return max(files, key=os.path.getsize) if files else None


def write_directions(ports_table, chunk_size=1_000_000):
    """
    Ποιος ξεκινά τη σύνδεση και ποιος ακούσει στη θύρα.

    Η σύνθεση των επαφών δείχνει ΜΕ ΠΟΙΟΝ μιλά ένα υποδίκτυο, όχι τι είναι.
    Αυτό το κρίνει η κατεύθυνση: μια διεύθυνση που την καλούν στη 9100 είναι
    εκτυπωτής, μια που καλεί τη 9100 είναι υπολογιστής που τυπώνει. Χωρίς
    αυτό το βήμα οι δύο ερμηνείες είναι ισοδύναμες.

    Διαβάζω ΜΙΑ ημέρα, τη μεγαλύτερη: η κατεύθυνση μιας υπηρεσίας δεν αλλάζει
    μέσα στην εβδομάδα, και τα clean_*.csv είναι 4,3 GB συνολικά.
    """
    path_in = busiest_clean_file()
    if path_in is None:
        print("    - [ΚΑΤΕΥΘΥΝΣΗ] Παραλείπεται: δεν βρέθηκαν clean_*.csv "
              "(τρέξε  python main.py --step clean).")
        return None

    interesting = sorted({int(p) for p in ports_table['Service_Port']})
    known = prefixes()
    print(f" -> Κατεύθυνση συνδέσεων από το {os.path.basename(path_in)} "
          f"για {len(interesting)} θύρες.")

    counts = {}
    columns = ['Src_IP', 'Src_Port', 'Dst_IP', 'Dst_Port']
    for chunk in pd.read_csv(path_in, usecols=columns, chunksize=chunk_size):
        source = label_subnets(chunk['Src_IP'], known).fillna(EXTERNAL)
        target = label_subnets(chunk['Dst_IP'], known).fillna(EXTERNAL)
        destination = pd.to_numeric(chunk['Dst_Port'], errors='coerce')
        for port in interesting:
            hit = destination == port
            if not hit.any():
                continue
            pairs = pd.Series(1, index=chunk.index[hit]).groupby(
                [source[hit], target[hit]]).sum()
            for (src, dst), n in pairs.items():
                key = (port, src, dst)
                counts[key] = counts.get(key, 0) + int(n)

    rows = []
    for (port, src, dst), n in counts.items():
        rows.append({'Θύρα': port, 'Ακούει': dst, 'Συνδέεται': src, 'Ροές': n})
    table = pd.DataFrame(rows)
    if table.empty:
        print("    - [ΚΑΤΕΥΘΥΝΣΗ] Καμία ροή προς τις θύρες της λίστας.")
        return None

    # Μερίδιο μέσα στο ζεύγος (θύρα, ακούει): ποιος καλεί αυτή την υπηρεσία.
    per_listener = table.groupby(['Θύρα', 'Ακούει'])['Ροές'].transform('sum')
    table['Ποσοστό_των_κλήσεων_%'] = (100 * table['Ροές'] / per_listener).round(2)
    table['Εικασία_υπηρεσίας'] = table['Θύρα'].map(PORT_NOTES).fillna('')
    table = table.sort_values(['Θύρα', 'Ροές'], ascending=[True, False])

    path_out = os.path.join(output_dir(), 'subnet_directions.csv')
    table.to_csv(path_out, index=False, encoding='utf-8-sig')
    print(f"    - [ΚΑΤΕΥΘΥΝΣΗ] {path_out}")
    # Μόνο οι γραμμές που λένε κάτι: ο ακροατής είναι υποδίκτυο της ανάλυσης.
    visible = table[table['Ακούει'] != EXTERNAL]
    if not visible.empty:
        print(visible.head(20).to_string(index=False))
    return table


def write_hourly(inside):
    """
    Ωριαίο προφίλ ανά υποδίκτυο.

    Δεν ξαναγράφω τον υπολογισμό: δανείζομαι τις ίδιες συναρτήσεις που
    χρησιμοποιεί το στάδιο hourly για τις ομάδες. Αν η κανονικοποίηση εδώ
    ήταν άλλη (π.χ. ποσοστό των συνολικών ροών αντί για ποσοστό ανά host),
    οι δύο καμπύλες θα φαίνονταν συγκρίσιμες στο ίδιο γράφημα χωρίς να
    είναι, που είναι χειρότερο από το να λείπει η καμπύλη.

    Οι μερικές ημέρες εξαιρούνται, για τον ίδιο λόγο που εξαιρούνται εκεί:
    λείπουν ώρες του 24ώρου, και αυτό ακριβώς μετράει η καμπύλη.
    """
    from src.models.hourly_profile import load_temporal, partial_days, hourly_shares

    temporal = load_temporal()
    if temporal is None:
        print("    - [ΩΡΙΑΙΟ] Παραλείπεται: δεν βρέθηκαν temporal_*.csv.")
        return None

    excluded, _ = partial_days(temporal)
    if excluded:
        print(f"    - [ΩΡΙΑΙΟ] Εξαιρούνται οι μερικές ημέρες: "
              f"{', '.join(str(d) for d in excluded)}")
        temporal = temporal[~temporal['Date'].isin(excluded)]
        if temporal.empty:
            print("    - [ΩΡΙΑΙΟ] Δεν έμεινε καμία πλήρης ημέρα.")
            return None

    all_labels = inside.set_index('Host')['Υποδίκτυο']
    written = {}
    for suffix, weekdays in (('all', False), ('weekdays', True)):
        data = temporal[temporal['DayOfWeek'] <= 4] if weekdays else temporal

        # Κατώφλι ροών. Η κανονικοποίηση ανά host δίνει ΙΣΟ βάρος σε κάθε host,
        # οπότε ένας host με δύο ροές που έτυχαν στην ίδια ώρα συνεισφέρει μια
        # κορυφή ύψους 100% στη μέση καμπύλη, ακριβώς όση και ένας host με ένα
        # εκατομμύριο ροές. Στο 192.168.1.x αυτό ακριβώς συνέβαινε: ένας host
        # με δύο ροές έφτιαχνε μόνος του τις 3,45 από τις 8,38 μονάδες της
        # κορυφής των 18:00, και η καμπύλη έδειχνε ωράριο εκεί που η εντροπία
        # ώρας έλεγε 0,998, δηλαδή σταθερό ρυθμό.
        #
        # Το στάδιο hourly δεν έχει τέτοιο κατώφλι και δεν το προσθέτω εκεί:
        # οι ομάδες του έχουν χιλιάδες hosts και ο θόρυβος εξομαλύνεται. Με
        # 26 ή 29 hosts δεν εξομαλύνεται.
        totals = data.groupby('Host')['Flows'].sum()
        enough = set(totals[totals >= SUBNET_MIN_FLOWS_HOURLY].index)
        labels = all_labels[all_labels.index.isin(enough)]
        dropped = len(all_labels) - len(labels)

        curve, _ = hourly_shares(data, labels)
        if curve is None:
            continue
        if dropped:
            print(f"    - [ΩΡΙΑΙΟ/{suffix}] Εκτός καμπύλης {dropped} hosts με "
                  f"λιγότερες από {SUBNET_MIN_FLOWS_HOURLY} ροές "
                  f"(δεν ορίζεται ωριαίο προφίλ σε τόσο λίγες)")
        path = os.path.join(output_dir(), f'subnet_hourly_{suffix}.csv')
        curve.round(6).to_csv(path, encoding='utf-8-sig')
        written[suffix] = curve
        print(f"    - [ΩΡΙΑΙΟ] {path}")

    if 'all' in written:
        curve = written['all']
        business = curve.loc[:, [h for h in curve.columns if 8 <= int(h) <= 17]].sum(axis=1)
        for name in curve.index:
            row = curve.loc[name]
            median = float(row.median())
            # Ο λόγος κορυφής προς διάμεσο είναι το νούμερο που ξεχωρίζει την
            # ευθεία από την καμπάνα: κοντά στο 1 σημαίνει επίπεδη, γύρω στο 2
            # σημαίνει ωράριο. Το Business_Ratio μόνο του δεν το λέει, αφού
            # μια επίπεδη καμπύλη δίνει ούτως ή άλλως 10/24 = 41,7%.
            ratio = float(row.max()) / median if median > 0 else float('nan')
            print(f"      {name}: εντός ωραρίου {100 * business[name]:.1f}%, "
                  f"κορυφή στις {int(row.idxmax()):02d}:00, "
                  f"λόγος κορυφή/διάμεσο {ratio:.2f}")
    return written or None


def read_separate_labels():
    """
    Οι ετικέτες της ΧΩΡΙΣΤΗΣ ομαδοποίησης κάθε υποδικτύου, όπως τις έγραψε το
    στάδιο micro.

    Δεν τις ξαναϋπολογίζω εδώ. Είχα δική μου υλοποίηση, που διάλεγε k με σκέτο
    silhouette και έγραφε δικά της CSV: δύο ομαδοποιήσεις του ίδιου πράγματος
    με διαφορετικά κριτήρια, που θα απέκλιναν στο πρώτο τρέξιμο όπου ο χρήστης
    απαντούσε άλλο k στην ερώτηση του kselect. Τώρα υπάρχει μία, περνά από
    τους πέντε δείκτες εγκυρότητας αντί για έναν, και η σύγκριση παρακάτω
    αφορά πραγματικά τις ετικέτες που μπαίνουν στην αναφορά.
    """
    labels = {}
    for key, title, _ in subnet_populations():
        path = os.path.join(PATHS['results_micro'], f'{key}_global.csv')
        name = title.replace('Υποδίκτυο ', '')
        if not os.path.exists(path):
            print(f"    - {name}: δεν βρέθηκε το {os.path.basename(path)}. "
                  f"Τρέξε πρώτα:  python main.py --step micro")
            continue
        frame = pd.read_csv(path, usecols=['Host', 'Cluster'])
        frame = frame[frame['Cluster'] >= 0]
        if frame.empty:
            continue
        labels[name] = frame.set_index('Host')['Cluster']
        sizes = sorted(frame['Cluster'].value_counts().tolist(), reverse=True)
        print(f"    - {name}: {len(frame)} hosts σε {len(sizes)} ομάδες, μεγέθη {sizes}")
    return labels


def cluster_joint(inside, k_range, separate):
    """
    Κοινή ομαδοποίηση όλων των υποδικτύων, και σύγκριση με τα δύο μέτρα
    που έχουν νόημα:

    α) ARI(ετικέτα, υποδίκτυο): αναπαράγει η συμπεριφορά τα σύνορα των
       διευθύνσεων; Αν ήταν κοντά στο 1, το υποδίκτυο θα ήταν ο πραγματικός
       διαχωριστής και η ομαδοποίηση θα μάθαινε τη διεύθυνση, όχι τη χρήση.
    β) ARI(κοινή, χωριστή) μέσα σε κάθε υποδίκτυο: βλέπει η κοινή
       ομαδοποίηση το ίδιο πράγμα με τη χωριστή; Όπου διαφέρουν, το
       υποδίκτυο είναι μικρό και οι ομάδες το διασπούν επειδή τις ορίζουν
       οι υπόλοιποι hosts.
    """
    matrix = scale_features(inside)
    truth = inside['Υποδίκτυο'].to_numpy()

    rows, labels_by_k = [], {}
    for k in k_range:
        if k >= len(inside):
            break
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                        n_init=10).fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        labels_by_k[k] = labels
        rows.append({
            'k': k,
            'Silhouette': round(float(silhouette_score(matrix, labels)), 4),
            'ARI_με_υποδίκτυο': round(float(adjusted_rand_score(truth, labels)), 4),
        })
    if not rows:
        print("    - Η κοινή ομαδοποίηση δεν έβγαλε έγκυρο αποτέλεσμα.")
        return

    table = pd.DataFrame(rows)
    path = os.path.join(output_dir(), 'subnet_ari.csv')
    table.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"    - [ARI] {path}")
    print(table.to_string(index=False))
    peak = table.loc[table['ARI_με_υποδίκτυο'].idxmax()]
    print(f"      Μέγιστο ARI με το υποδίκτυο: {peak['ARI_με_υποδίκτυο']:.3f} "
          f"στο k={int(peak['k'])}. Όσο πιο κοντά στο 0, τόσο λιγότερο "
          f"ακολουθεί η συμπεριφορά τη διεύθυνση.")

    # Σταυροπίνακας και συμφωνία για ΚΑΘΕ k που δοκιμάστηκε, όχι μόνο για ένα
    # «καλύτερο». Και τα δύο εξαρτώνται από το k της κοινής ομαδοποίησης: στο
    # k=2 ο σταυροπίνακας έχει δύο γραμμές και δεν λέει τίποτα, στο k=5 δείχνει
    # ότι κάθε ομάδα περιέχει διευθύνσεις από δύο ή τρία υποδίκτυα. Μία μόνο
    # τιμή κρύβει αυτή την εξάρτηση και προσκαλεί σε cherry-picking. Τυπώνω
    # του καλύτερου silhouette, γράφω όλα.
    # Τυπώνω το k που ΕΥΝΟΕΙ περισσότερο την υπόθεση που ελέγχεται, δηλαδή
    # το μεγαλύτερο ARI προς το υποδίκτυο, και όχι το καλύτερο silhouette.
    # Αν ούτε στην πιο ευνοϊκή του μορφή η κοινή ομαδοποίηση δεν αναπαράγει
    # τα υποδίκτυα, το συμπέρασμα δεν είναι τεχνούργημα κακής επιλογής k.
    # Με το silhouette το k έβγαινε 2, που δίνει σταυροπίνακα δύο γραμμών
    # και δεν δείχνει απολύτως τίποτα.
    best_k = int(table.loc[table['ARI_με_υποδίκτυο'].idxmax(), 'k'])
    cross_rows, agree_rows = [], []
    for k, labels in labels_by_k.items():
        joint = pd.Series(labels, index=inside['Host'], name='Ετικέτα')
        crosstab = pd.crosstab(joint.to_numpy(), inside['Υποδίκτυο'])
        for label, counts in crosstab.iterrows():
            row = {'k': k, 'Ετικέτα': int(label)}
            row.update({name: int(value) for name, value in counts.items()})
            cross_rows.append(row)

        for name, separate_labels in separate.items():
            common = joint[joint.index.isin(separate_labels.index)]
            aligned = separate_labels.reindex(common.index)
            groups_here = int(pd.Series(common.to_numpy()).nunique())
            agree_rows.append({
                'Υποδίκτυο': name,
                'k_κοινής': k,
                'Hosts': len(aligned),
                'Ομάδες_χωριστά': int(aligned.nunique()),
                'Ομάδες_της_κοινής_εδώ': groups_here,
                # Όταν η κοινή ομαδοποίηση βάζει ΟΛΟ το υποδίκτυο σε μία
                # ομάδα, το ARI βγαίνει 0 εξ ορισμού και διαβάζεται λάθος ως
                # «πλήρης διαφωνία»: δεν διαφωνεί, απλώς δεν διέσπασε τίποτα.
                # Γράφω κενό, ώστε να μη μπει στην αναφορά ένα μηδενικό που
                # δεν σημαίνει αυτό.
                'ARI_κοινή_vs_χωριστή': (
                    round(float(adjusted_rand_score(aligned.to_numpy(),
                                                    common.to_numpy())), 4)
                    if groups_here > 1 else np.nan),
            })

    crosstabs = pd.DataFrame(cross_rows).fillna(0)
    path = os.path.join(output_dir(), 'subnet_vs_cluster.csv')
    crosstabs.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"    - [ΣΤΑΥΡΟΠΙΝΑΚΑΣ] {path}  (όλα τα k· παρακάτω το k={best_k})")
    print(crosstabs[crosstabs['k'] == best_k].to_string(index=False))

    agreement = pd.DataFrame(agree_rows).sort_values(['Υποδίκτυο', 'k_κοινής'])
    path = os.path.join(output_dir(), 'subnet_agreement.csv')
    agreement.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"    - [ΣΥΜΦΩΝΙΑ] {path}  (όλα τα k· παρακάτω το k={best_k})")
    print(agreement[agreement['k_κοινής'] == best_k].to_string(index=False))


def run(k_range=None):
    """
    Ανάλυση των ιδιωτικών υποδικτύων χωριστά.

    Δεν αγγίζει τίποτα από τα υπάρχοντα αποτελέσματα: γράφει μόνο στο
    results/subnets/ και δεν αλλάζει το INTERNAL_SUBNET, οπότε τα macro και
    micro μένουν όπως είναι. Είναι διερεύνηση του ερωτήματος «τι είναι αυτά
    τα υποδίκτυα και πρέπει να μπουν μέσα στο εσωτερικό δίκτυο;», όχι
    απάντησή του: την απάντηση τη δίνει το εργαστήριο.
    """
    print("\n--- Έναρξη: Ανάλυση Ιδιωτικών Υποδικτύων ---")
    # Το k_range αφορά ΜΟΝΟ την κοινή ομαδοποίηση των τριών υποδικτύων μαζί,
    # που δεν αντιστοιχεί σε κανέναν πληθυσμό του pipeline και άρα δεν έχει
    # δικούς της δείκτες. Το k κάθε υποδικτύου χωριστά το διάλεξε το kselect.
    k_range = list(SUBNET_K_RANGE if k_range is None else k_range)
    print(f" -> Υποδίκτυα: {', '.join(group_label(p) for p in prefixes())}")
    print(f" -> Εύρος k: {min(k_range)}-{max(k_range)}")

    every, inside = labelled_features()
    if inside.empty:
        print("[!] Κανένας host στα υποδίκτυα της ανάλυσης. Έλεγξε το SUBNET_PREFIXES.")
        return

    write_inventory(every, inside)
    write_profiles(inside)

    bidir = bidir_frame()
    if bidir is None:
        print("    - Παραλείπονται επαφές, θύρες και κατεύθυνση: δεν βρέθηκαν bidir_*.csv.")
    else:
        write_peers(bidir)
        ports = write_ports(bidir)
        write_directions(ports)
    write_hourly(inside)

    print("\n -> Ετικέτες της χωριστής ομαδοποίησης (από το στάδιο micro):")
    separate = read_separate_labels()
    print("\n -> Κοινή ομαδοποίηση και σύγκριση:")
    cluster_joint(inside, k_range, separate)

    print("--- Ολοκληρώθηκε ---")


if __name__ == "__main__":
    run()
