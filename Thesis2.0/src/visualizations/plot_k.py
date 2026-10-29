# src/visualizations/plot_k.py
"""
Οι καμπύλες των δεικτών εγκυρότητας ανά k, ένα γράφημα για κάθε πληθυσμό.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ. Η επιλογή του k είναι η μόνη παράμετρος της εργασίας που δεν
προκύπτει από τα δεδομένα μόνη της: έξι κριτήρια δίνουν έξι απαντήσεις και η
τελική τιμή είναι συμβιβασμός. Ο πίνακας του kselect δείχνει τους αριθμούς,
αλλά ο αριθμός δεν δείχνει το σχήμα — αν η κορυφή του Silhouette είναι αιχμή ή
πλατό, αν το Davies-Bouldin πέφτει μονότονα (οπότε η «βέλτιστη» τιμή του είναι
απλώς η άκρη του εύρους), αν το γόνατο της αδράνειας είναι γόνατο ή ευθεία. Το
σχήμα είναι η δικαιολόγηση του συμβιβασμού, και χωρίς γράφημα δεν φαίνεται.

ΜΙΑ ΠΗΓΗ ΑΛΗΘΕΙΑΣ. Τίποτα εδώ δεν υπολογίζεται ξανά. Οι καμπύλες διαβάζονται
από τα k_selection_<level>.csv που έγραψε το kselect και οι προτάσεις των
κριτηρίων βγαίνουν από την ΙΔΙΑ votes() που τύπωσε τον πίνακα στην οθόνη. Αν
κάποτε αλλάξει ο κανόνας ενός κριτηρίου, αλλάζει σε ένα σημείο και το γράφημα
ακολουθεί. Μια δεύτερη υλοποίηση εδώ θα μπορούσε να δείχνει άλλο k από εκείνο
που είδε ο χρήστης όταν αποφάσισε, που είναι ο χειρότερος τρόπος να λέει
ψέματα ένα γράφημα.

ΠΟΙΟΙ ΠΛΗΘΥΣΜΟΙ. Όσοι έχουν αρχείο, όχι όσοι είναι γραμμένοι σε μια λίστα εδώ.
Τα δύο ορφανά PNG που βρέθηκαν στο results/k_selection/ είχαν προκύψει
ακριβώς από το αντίθετο: παλιός κώδικας σχεδίαζε δύο συγκεκριμένα επίπεδα,
ο αριθμός των πληθυσμών άλλαξε, και οι εικόνες έμειναν να δείχνουν μια
κατάσταση που δεν υπήρχε πια. Με τη σάρωση του φακέλου αυτό δεν
ξανασυμβαίνει: ό,τι μετρήθηκε σχεδιάζεται, ό,τι δεν μετρήθηκε λείπει φανερά.
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.config import BASE_DIR
from src.models.populations import default_k
from src.models.select_k import LEVELS, output_dir, load_k_table, votes

# (στήλη, τίτλος, όνομα του κριτηρίου στο votes(), στήλη τυπικής απόκλισης,
#  προς ποια κατεύθυνση είναι «καλύτερα»)
#
# Η σειρά είναι η σειρά του πίνακα της αναφοράς, ώστε να διαβάζονται μαζί.
CRITERIA = [
    ('Inertia',           'Αδράνεια (WCSS)',   'Elbow',             None,
     'γόνατο'),
    ('Silhouette',        'Silhouette',        'Silhouette',        'Sil_sd',
     'ψηλότερα'),
    ('Calinski_Harabasz', 'Calinski–Harabasz', 'Calinski-Harabasz', None,
     'ψηλότερα'),
    ('Davies_Bouldin',    'Davies–Bouldin',    'Davies-Bouldin',    None,
     'χαμηλότερα'),
    ('Gap',               'Gap statistic',     'Gap statistic',     'Gap_s',
     'κανόνας Tibshirani'),
    ('Stability_ARI',     'Ευστάθεια (ARI)',   'Ευστάθεια ARI',     'Stab_sd',
     'ψηλότερα'),
]

CURVE = '#33506b'        # σκούρο μπλε-γκρι: διαβάζεται και σε ασπρόμαυρη εκτύπωση
PICK = '#c0392b'         # η πρόταση του κριτηρίου
CHOSEN = '#1f8a4c'       # το k που χρησιμοποιήθηκε


def levels_found(folder=None):
    """
    Τα επίπεδα που έχουν πίνακα δεικτών στον δίσκο, με σταθερή σειρά.

    Η σειρά είναι εκείνη των LEVELS (macro, micro, υποδίκτυα) και ό,τι βρεθεί
    εκτός τους μπαίνει στο τέλος αλφαβητικά — εκεί πέφτει το macro_dedup, που
    δεν είναι πληθυσμός της ανάλυσης αλλά έλεγχος πάνω στον macro.
    """
    folder = folder or output_dir()
    keys = {os.path.basename(p)[len('k_selection_'):-len('.csv')]
            for p in glob.glob(os.path.join(folder, 'k_selection_*.csv'))}
    known = [key for key in LEVELS if key in keys]
    return known + sorted(keys - set(known))


def level_title(level):
    """Ο τίτλος του επιπέδου. Τα LEVELS τον έχουν· το macro_dedup όχι."""
    if level in LEVELS:
        return LEVELS[level][0]
    if level == 'macro_dedup':
        return 'Μακροσκοπική ανάλυση, μοναδικά προφίλ (έλεγχος)'
    return level


def chosen_k(level):
    """
    Το k που χρησιμοποιήθηκε, και από πού ήρθε.

    Προτεραιότητα στο K_USED του kselect, που είναι αυτό που όντως συνέβη σε
    αυτή την εκτέλεση — ακόμη και αν ο χρήστης απάντησε διαφορετικά από το
    config στη διαδραστική ερώτηση. Αν το στάδιο δεν έτρεξε στην ίδια
    διεργασία (π.χ.  --step graphs), πέφτει στο config, που είναι ό,τι θα
    χρησιμοποιούσε μια μη διαδραστική εκτέλεση. Γυρίζει (None, None) για
    επίπεδα που δεν είναι πληθυσμοί, όπως το macro_dedup.
    """
    from src.models.select_k import K_USED
    if level in K_USED:
        return K_USED[level]['k'], K_USED[level]['source']
    if level in LEVELS:
        return default_k(level), 'config'
    return None, None


def panel(ax, df, column, title, pick, sd_column, direction, k_used):
    """Μία καμπύλη: οι τιμές, η πρόταση του κριτηρίου, το k που επιλέχθηκε."""
    ks = df['k'].to_numpy(dtype=float)
    values = df[column].to_numpy(dtype=float)

    if k_used is not None:
        ax.axvline(k_used, color=CHOSEN, lw=1.6, ls='--', zorder=1)

    if sd_column and sd_column in df.columns:
        sd = df[sd_column].to_numpy(dtype=float)
        ax.fill_between(ks, values - sd, values + sd, color=CURVE, alpha=0.15, lw=0)
    ax.plot(ks, values, marker='o', ms=4.5, lw=1.8, color=CURVE, zorder=2)

    # Η πρόταση του κριτηρίου: κύκλος χωρίς γέμισμα, για να ξεχωρίζει και
    # ασπρόμαυρα από τα κανονικά σημεία της καμπύλης.
    if pick is not None and pick in set(df['k']):
        y = float(df.loc[df['k'] == pick, column].iloc[0])
        ax.scatter([pick], [y], s=190, facecolor='none', edgecolor=PICK,
                   linewidth=2.2, zorder=3)

    if pick is None:
        label = 'δεν σταθεροποιείται'
    elif pick in (int(ks.min()), int(ks.max())):
        # Όταν η καμπύλη είναι μονότονη, το «βέλτιστο» του κριτηρίου δεν είναι
        # κορυφή αλλά το σημείο όπου σταμάτησε η δοκιμή. Χωρίς αυτή τη σήμανση
        # ο αναγνώστης το μετράει ως κανονική ψήφο.
        label = f'k={pick} — άκρο του εύρους που δοκιμάστηκε'
    else:
        label = f'k={pick}'
    ax.set_title(f'{title}\n{direction} · το κριτήριο δείχνει {label}', fontsize=10.5)
    ax.set_xlabel('k')
    ax.set_xticks(df['k'].astype(int))
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)


def plot_level(level, graphs_dir):
    """Το γράφημα ενός επιπέδου. False αν λείπει ή είναι άκυρος ο πίνακας."""
    df = load_k_table(level)
    if df is None:
        return False
    df = df.sort_values('k')
    missing = [column for column, *_ in CRITERIA if column not in df.columns]
    if missing:
        print(f" -> Ο πίνακας του «{level}» δεν έχει τις στήλες "
              f"{', '.join(missing)}· παραλείπεται.")
        return False

    k_used, source = chosen_k(level)
    picks = votes(df)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    for ax, (column, title, name, sd_column, direction) in zip(axes.ravel(), CRITERIA):
        panel(ax, df, column, title, picks.get(name), sd_column, direction, k_used)

    # ΠΡΟΣΟΧΗ ΣΤΗ ΔΙΑΤΥΠΩΣΗ. Σκέτο «το δείχνουν 2 από τα 6» διαβάζεται ως
    # καταδίκη της επιλογής, επειδή υπονοεί ότι σωστό k είναι αυτό που
    # ψηφίζουν τα περισσότερα κριτήρια. Ο κανόνας της εργασίας δεν είναι
    # πλειοψηφία αλλά συμβιβασμός: ζητείται το k που χάνει όσο το δυνατόν
    # λιγότερο από καθένα. Το εύρος και η διάμεσος των προτάσεων δείχνουν αν
    # υπήρχε καν k που να τα ικανοποιεί όλα — συνήθως δεν υπάρχει.
    agree = sum(1 for _, _, name, _, _ in CRITERIA if picks.get(name) == k_used)
    spread = sorted(k for k in picks.values() if k is not None)
    if k_used is None:
        note = 'χωρίς επιλεγμένο k: δεν είναι πληθυσμός της ανάλυσης'
    elif spread:
        note = (f'k που χρησιμοποιήθηκε: {k_used} (πηγή: {source})  ·  '
                f'οι προτάσεις των κριτηρίων απλώνονται σε {spread[0]}–{spread[-1]} '
                f'με διάμεσο {int(np.median(spread))}  ·  '
                f'ακριβώς στο {k_used} δείχνουν {agree} από τα {len(CRITERIA)}')
    else:
        note = f'k που χρησιμοποιήθηκε: {k_used} (πηγή: {source})'
    fig.suptitle(f'Δείκτες εγκυρότητας ανά k — {level_title(level)}\n{note}',
                 fontsize=13, y=1.01)

    # Υπόμνημα μία φορά για όλο το γράφημα, με χειροποίητους δείκτες: τα
    # στοιχεία είναι τα ίδια σε κάθε υπογράφημα και έξι αντίγραφα του ίδιου
    # υπομνήματος δεν προσθέτουν πληροφορία.
    handles = [
        Line2D([], [], color=CURVE, marker='o', ms=5, lw=1.8, label='τιμή του δείκτη'),
        Line2D([], [], color=PICK, marker='o', ms=12, lw=0, markerfacecolor='none',
               markeredgewidth=2.2, label='η πρόταση αυτού του κριτηρίου'),
    ]
    if k_used is not None:
        handles.append(Line2D([], [], color=CHOSEN, lw=1.6, ls='--',
                              label=f'το k που χρησιμοποιήθηκε ({k_used})'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.035))

    # Το rect κρατά χώρο για τον suptitle πάνω και το υπόμνημα κάτω· χωρίς
    # αυτό το tight_layout τα θεωρεί ανύπαρκτα και τα γράφει πάνω στα πλαίσια.
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out_img = os.path.join(graphs_dir, f'k_criteria_{level}.png')
    fig.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" -> [ΕΠΙΤΥΧΙΑ] Το γράφημα αποθηκεύτηκε στο: {out_img}")
    return True


def run(graphs_dir=None):
    """Ένα γράφημα ανά πληθυσμό. True αν σχεδιάστηκε έστω ένα."""
    if graphs_dir is None:
        graphs_dir = os.path.join(BASE_DIR, 'results', 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)

    levels = levels_found()
    if not levels:
        print(" -> Δεν βρέθηκαν πίνακες δεικτών "
              "(τρέξε  python main.py --step kselect).")
        return False

    # ΟΧΙ any(generator): θα σταματούσε στο πρώτο επίπεδο που σχεδιάστηκε και
    # τα υπόλοιπα δεν θα έβγαιναν ποτέ.
    drawn = [plot_level(level, graphs_dir) for level in levels]
    return any(drawn)


if __name__ == "__main__":
    run()
