# src/models/ablation.py
"""
Ποια οικογένεια χαρακτηριστικών κρατά την ομαδοποίηση όρθια.

ΤΟ ΕΡΩΤΗΜΑ. Η εργασία ισχυρίζεται ότι τα χρονικά χαρακτηριστικά είναι αυτά που
κάνουν δυνατή την απόδοση ρόλων — ότι χωρίς αυτά οι hosts δεν ξεχωρίζουν σε
σταθμούς εργασίας, servers και συσκευές συνεχούς λειτουργίας. Μέχρι τώρα αυτό
ήταν ισχυρισμός που στηριζόταν στην ανάγνωση των ωριαίων καμπυλών. Εδώ
μετριέται: αφαιρείται μια οικογένεια κάθε φορά, ξανατρέχει η ίδια ομαδοποίηση,
και συγκρίνεται με ARI η νέα διαμέριση με την πλήρη.

ΓΙΑΤΙ Η ΣΥΓΚΡΙΣΗ ΕΙΝΑΙ ΔΙΚΑΙΗ. Τα δεκαπέντε χαρακτηριστικά χωρίζονται σε τρεις
οικογένειες των **πέντε**. Αυτό δεν είναι σύμπτωση που βολεύει, είναι ο λόγος
που το πείραμα λέει κάτι: αν τα χρονικά ήταν επτά και τα θυρών τρία, το «χωρίς
τα χρονικά χαλάει περισσότερο» θα μπορούσε να σημαίνει απλώς ότι αφαιρέθηκαν
περισσότερες διαστάσεις. Με ίσες οικογένειες και ίδιο k, η διαφορά είναι της
πληροφορίας και όχι του πλήθους.

ΤΙ ΔΕΝ ΜΕΤΡΑΕΙ. Το ARI λέει πόσο αναπαράγεται η ΔΙΑΜΕΡΙΣΗ, όχι πόσο σωστή
είναι η ερμηνεία της. Ότι μια ομάδα είναι «σταθμοί εργασίας» εξακολουθεί να
στηρίζεται στις ωριαίες καμπύλες και στις θύρες, και χωρίς ground truth μένει
υπόθεση. Αυτό που κλείνει εδώ είναι το ενδιάμεσο βήμα: αν η δομή που
ερμηνεύεται οφείλεται όντως στα χρονικά χαρακτηριστικά ή αν θα υπήρχε και
χωρίς αυτά.

Η ΠΛΗΡΗΣ ΠΑΡΑΛΛΑΓΗ ΩΣ ΕΛΕΓΧΟΣ. Η γραμμή «όλα» ξανατρέχει την ομαδοποίηση με τα
δεκαπέντε και τη συγκρίνει με τις ετικέτες που είναι ήδη γραμμένες στον δίσκο.
Πρέπει να βγει ARI = 1,000. Αν δεν βγει, κάτι δεν αναπαράγεται — διαφορετικό
scaling, διαφορετικός σπόρος, διαφορετικό k — και τότε κανένα από τα υπόλοιπα
νούμερα του πίνακα δεν σημαίνει τίποτα.
"""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from src.config import (PATHS, BASE_DIR, RANDOM_STATE, FEATURE_COLUMNS,
                        LOG_FEATURES, LINEAR_FEATURES, RATIO_FEATURES)
from src.utils.scaling import scale_features
from src.models.populations import populations, default_k, micro_title
from src.models.select_k import silhouette_sampled

# Τρεις οικογένειες των πέντε. Η κατανομή δεν ακολουθεί τις λίστες του config
# (LOG/LINEAR/RATIO), που είναι τεχνικές — λένε ποιος μετασχηματισμός
# εφαρμόζεται, όχι τι μετράει το χαρακτηριστικό. Εδώ η ομαδοποίηση είναι
# σημασιολογική: πόσο, πότε, και προς τα πού.
GROUPS = {
    'χρονικά': ['Active_Hours', 'Active_Days', 'Hour_Entropy',
                'Business_Ratio', 'Weekend_Ratio'],
    'όγκου': ['Bytes_In', 'Bytes_Out', 'Total_Flows', 'Unique_Peers',
              'Avg_Bytes_Per_Flow'],
    'θυρών': ['Port_Entropy', 'Out_Ratio', 'TCP_Ratio', 'Well_Known_Ratio',
              'P2P_Port_Ratio'],
}


def check_groups():
    """
    Οι τρεις οικογένειες πρέπει να είναι ΑΚΡΙΒΩΣ τα FEATURE_COLUMNS.

    Αν κάποτε προστεθεί χαρακτηριστικό στο config και ξεχαστεί εδώ, το πείραμα
    θα έτρεχε κανονικά και θα έβγαζε πίνακα — απλώς η «πλήρης» παραλλαγή δεν
    θα ήταν πλήρης και κάθε σύγκριση θα ήταν λάθος κατά ένα χαρακτηριστικό,
    σιωπηλά. Προτιμώ να σκάσει.
    """
    mine = [name for names in GROUPS.values() for name in names]
    extra = sorted(set(mine) - set(FEATURE_COLUMNS))
    missing = sorted(set(FEATURE_COLUMNS) - set(mine))
    if extra or missing:
        raise ValueError(
            "Οι οικογένειες χαρακτηριστικών του ablation δεν συμφωνούν με τα "
            f"FEATURE_COLUMNS. Λείπουν: {missing or '—'}. "
            f"Περισσεύουν: {extra or '—'}.")
    if len(mine) != len(set(mine)):
        raise ValueError("Χαρακτηριστικό σε δύο οικογένειες ταυτόχρονα.")


def variants():
    """(όνομα, χαρακτηριστικά) για κάθε παραλλαγή, με την πλήρη πρώτη."""
    out = [('όλα', list(FEATURE_COLUMNS))]
    for name, members in GROUPS.items():
        out.append((f'χωρίς {name}',
                    [c for c in FEATURE_COLUMNS if c not in members]))
    for name, members in GROUPS.items():
        out.append((f'μόνο {name}', list(members)))
    return out


def scale_subset(df, columns):
    """Κλιμάκωση μόνο των επιλεγμένων στηλών, με τον ίδιο κανόνα ανά τύπο."""
    keep = set(columns)
    return scale_features(
        df,
        log_columns=[c for c in LOG_FEATURES if c in keep],
        linear_columns=[c for c in LINEAR_FEATURES if c in keep],
        ratio_columns=[c for c in RATIO_FEATURES if c in keep],
    )


def sources():
    """
    (κλειδί, τίτλος, αρχείο) για κάθε πληθυσμό που έχει ομαδοποίηση.

    Ο macro μπαίνει με το χέρι γιατί δεν είναι στο populations(): ορίζεται
    αφαιρετικά και περνά από Isolation Forest πριν την ομαδοποίηση, οπότε το
    αρχείο του είναι οι ΚΑΘΑΡΟΙ hosts και όχι ο πληθυσμός όπως βγήκε από τον
    διαχωρισμό.
    """
    found = [('macro', 'Μακροσκοπική ανάλυση (καθαροί εξωτερικοί)',
              os.path.join(PATHS['results_macro'], 'clean_macro_global.csv'))]
    for key, title, _ in populations():
        found.append((key, micro_title() if key == 'micro' else title,
                      os.path.join(PATHS['results_micro'], f'{key}_global.csv')))
    return found


def analyse(key, title, path):
    """Οι παραλλαγές ενός πληθυσμού. Λίστα από dict, μία ανά παραλλαγή."""
    df = pd.read_csv(path)
    if 'Cluster' not in df.columns:
        print(f" -> Το {os.path.basename(path)} δεν έχει στήλη Cluster· "
              "παραλείπεται.")
        return []

    reference = df['Cluster'].to_numpy()
    k = default_k(key)

    # Το k του config πρέπει να είναι το ίδιο με τις ομάδες που υπάρχουν στο
    # αρχείο. Αν διαφέρουν, το αρχείο είναι από άλλη εκτέλεση και η σύγκριση
    # θα μετρούσε τη διαφορά των δύο εκτελέσεων αντί για το ablation.
    present = int(len(np.unique(reference)))
    if present != k:
        print(f" -> [!] Το {os.path.basename(path)} έχει {present} ομάδες ενώ "
              f"το config λέει k={k}. Ξανατρέξε την ομαδοποίηση· "
              "παραλείπεται.")
        return []
    print(f"\n  {title}  —  {len(df):,} hosts, k = {k}")

    rows = []
    for name, columns in variants():
        X = scale_subset(df, columns)
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                        n_init=10).fit_predict(X)
        ari = adjusted_rand_score(reference, labels)
        sil, _ = silhouette_sampled(X, labels)
        rows.append({
            'Πληθυσμός': title, 'Κλειδί': key, 'Hosts': len(df), 'k': k,
            'Παραλλαγή': name, 'Χαρακτηριστικά': len(columns),
            'ARI_με_πλήρη': round(float(ari), 4),
            'Silhouette': round(float(sil), 4) if np.isfinite(sil) else None,
        })
        flag = ''
        if name == 'όλα' and ari < 0.999:
            flag = '   <-- ΔΕΝ ΑΝΑΠΑΡΑΓΕΤΑΙ'
        print(f"    {name:<16} {len(columns):>2} χαρ.   "
              f"ARI {ari:6.3f}   Silhouette {sil:6.3f}{flag}")
    return rows


# Πάνω από αυτό το ARI θεωρώ ότι η διαμέριση «αναπαράγεται». Είναι σύμβαση
# ανάγνωσης και όχι στατιστικό τεστ· μπαίνει εδώ ως σταθερά ώστε να φαίνεται
# ότι είναι επιλογή, και οι δύο αριθμοί τυπώνονται πάντα για να κρίνει ο
# αναγνώστης μόνος του.
CLEAR = 0.70


def commentary(table):
    """
    Η ανάγνωση του πίνακα, ανά πληθυσμό και ανά οικογένεια.

    Δύο ερωτήματα, που ΔΕΝ είναι το ίδιο και η σύγχυσή τους ήταν λάθος στην
    πρώτη εκδοχή αυτής της συνάρτησης:

      ΕΠΑΡΚΕΙΑ  — ARI της «μόνο X»: φτάνει η οικογένεια μόνη της;
      ΑΝΑΓΚΑΙΟΤΗΤΑ — ARI της «χωρίς X»: χαλάει η δομή όταν φύγει;

    Μια οικογένεια μπορεί να είναι επαρκής και ΜΗ αναγκαία: τότε η πληροφορία
    της υπάρχει και αλλού, και η σωστή λέξη είναι «πλεονάζουσα», όχι
    «μοιρασμένη». Στο 192.168.1.x συμβαίνει ακριβώς αυτό — ο όγκος μόνος του
    δίνει 0,929 ενώ η αφαίρεσή του δεν κοστίζει τίποτα (1,000) — και η πρώτη
    εκδοχή το ανέφερε ως «καμία οικογένεια δεν αρκεί», που είναι το αντίθετο
    από ό,τι λένε τα νούμερα.
    """
    print("\n" + "=" * 72)
    print(f"  ΑΝΑΓΝΩΣΗ  (όριο «αναπαράγεται»: ARI > {CLEAR:.2f})")
    print("=" * 72)
    for key in table['Κλειδί'].unique():
        part = table[table['Κλειδί'] == key].set_index('Παραλλαγή')
        print(f"\n  {part['Πληθυσμός'].iloc[0]}")
        for family in GROUPS:
            enough = part.loc[f'μόνο {family}', 'ARI_με_πλήρη']
            without = part.loc[f'χωρίς {family}', 'ARI_με_πλήρη']

            if enough > CLEAR and without <= CLEAR:
                verdict = 'ΚΡΑΤΑΕΙ ΤΗ ΔΟΜΗ (επαρκής και αναγκαία)'
            elif enough > CLEAR:
                verdict = 'πλεονάζουσα: αρκεί μόνη της, αλλά υπάρχει και αλλού'
            elif without <= CLEAR:
                verdict = 'αναγκαία αλλά όχι επαρκής'
            else:
                verdict = 'ούτε αρκεί μόνη της ούτε λείπει όταν φύγει'
            print(f"    {family:<10} μόνο {enough:5.3f} | χωρίς {without:5.3f}"
                  f"   {verdict}")


def run():
    print("\n--- Έναρξη: Ablation Χαρακτηριστικών ---")
    check_groups()
    print(f" -> {len(FEATURE_COLUMNS)} χαρακτηριστικά σε {len(GROUPS)} "
          f"οικογένειες των {len(next(iter(GROUPS.values())))}")
    print(" -> Ίδιο k, ίδιος σπόρος, ίδια κλιμάκωση· αλλάζουν μόνο οι στήλες.")

    rows = []
    for key, title, path in sources():
        if not os.path.exists(path):
            print(f"\n  [!] Δεν βρέθηκε το {path} — παραλείπεται ο «{title}».")
            continue
        rows.extend(analyse(key, title, path))

    if not rows:
        print("\n -> Κανένας πληθυσμός δεν είχε αρχείο ομαδοποίησης. "
              "Τρέξε πρώτα:  python main.py --step macro micro")
        return

    table = pd.DataFrame(rows)
    commentary(table)

    folder = os.path.join(BASE_DIR, 'results', 'ablation')
    os.makedirs(folder, exist_ok=True)
    output = os.path.join(folder, 'ablation.csv')
    table.to_csv(output, index=False, encoding='utf-8-sig')
    print(f"\n -> Ο πίνακας αποθηκεύτηκε στο: {output}")
    print("--- Ολοκληρώθηκε ---")


if __name__ == "__main__":
    run()
