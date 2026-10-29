# src/visualizations/plot_subnets.py
"""
Τα γραφήματα του σταδίου 'subnets'.

Τέσσερα γραφήματα, ένα για κάθε ισχυρισμό που κάνει η ανάλυση:

  subnets_peers.png     πού πάει η κίνηση κάθε υποδικτύου  -> δεν είναι hosts
                        του διαδικτύου
  subnets_hourly.png    η κατανομή μέσα στο 24ωρο          -> μηχανές και όχι
                        άνθρωποι
  subnets_k.png         silhouette και ARI ανά k           -> η συμπεριφορά δεν
                        ακολουθεί τη διεύθυνση

Δεν φτιάχνω γράφημα για τις θύρες: εκεί η πληροφορία είναι ότι ΜΙΑ θύρα
σηκώνει το 80% ή το 95% της κίνησης, και αυτό ένας πίνακας τεσσάρων γραμμών το
λέει καλύτερα από ράβδους.

Ούτε για τις ίδιες τις ομάδες κάθε υποδικτύου. Από τη στιγμή που τα υποδίκτυα
έγιναν κανονικοί πληθυσμοί του micro_profiling, τα scatter τους τα βγάζει το
plot_results με UMAP, με τον ίδιο ακριβώς κώδικα που βγάζει τα macro και
micro. Μια δεύτερη υλοποίηση εδώ θα έσπαγε τη συνοχή της εργασίας χωρίς λόγο.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import SUBNET_MIN_FLOWS_HOURLY, BASE_DIR, INTERNAL_SUBNET
from src.models.populations import subnet_populations
from src.models.subnet_analysis import EXTERNAL, output_dir, group_label, prefixes

# Η τιμή που θα είχε κάθε ώρα αν η κίνηση ήταν ομοιόμορφη στο 24ωρο. Ίδια
# γραμμή αναφοράς με το plot_results, ώστε τα δύο ωριαία γραφήματα να
# διαβάζονται το ένα δίπλα στο άλλο.
UNIFORM_SHARE = 100 / 24


def subnet_order():
    """Τα υποδίκτυα με σταθερή σειρά, το δηλωμένο εσωτερικό πρώτο."""
    return [group_label(prefix) for prefix in prefixes()]


def subnet_colors():
    """
    Ένα χρώμα ανά υποδίκτυο, ΤΟ ΙΔΙΟ σε όλα τα γραφήματα.

    Αν το 192.168.8.x ήταν πορτοκαλί στο ένα γράφημα και πράσινο στο άλλο, ο
    αναγνώστης θα έπρεπε να διαβάζει υπόμνημα σε κάθε εικόνα.
    """
    palette = sns.color_palette('tab10', n_colors=max(len(subnet_order()), 3))
    colors = {name: palette[i] for i, name in enumerate(subnet_order())}
    colors[EXTERNAL] = '0.65'
    return colors


def present(order, available):
    """Η σειρά του order, κρατώντας μόνο ό,τι υπάρχει στα δεδομένα."""
    return [name for name in order if name in set(available)]


def save(fig, filename, graphs_dir):
    path = os.path.join(graphs_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" -> [ΕΠΙΤΥΧΙΑ] Το γράφημα αποθηκεύτηκε στο: {path}")
    return True


def plot_peers(graphs_dir):
    """
    Πού πάει η κίνηση κάθε υποδικτύου, ως ποσοστό των ροών του.

    Σωρευτικές ράβδοι και όχι ομαδοποιημένες: το ερώτημα είναι πώς μοιράζεται
    το σύνολο της κίνησης ενός υποδικτύου, δηλαδή μέρη ενός όλου, και η
    σωρευτική ράβδος δείχνει ακριβώς αυτό σε μία ματιά. Το «εκτός των τριών»
    μπαίνει τελευταίο, γιατί αυτό είναι το νούμερο που κρίνει το ερώτημα.
    """
    path = os.path.join(output_dir(), 'subnet_peers.csv')
    if not os.path.exists(path):
        return False
    table = pd.read_csv(path, encoding='utf-8-sig')
    if table.empty:
        return False

    wide = table.pivot(index='Υποδίκτυο', columns='Μιλά_με',
                       values='Ποσοστό_του_υποδικτύου_%').fillna(0)
    rows = present(subnet_order(), wide.index)
    segments = present(subnet_order(), wide.columns) + (
        [EXTERNAL] if EXTERNAL in wide.columns else [])
    if not rows or not segments:
        return False
    wide = wide.loc[rows, segments]
    colors = subnet_colors()

    fig, ax = plt.subplots(figsize=(11, 1.25 * len(rows) + 2.2))
    left = np.zeros(len(rows))
    for segment in segments:
        values = wide[segment].to_numpy(dtype=float)
        ax.barh(rows, values, left=left, color=colors.get(segment, '0.8'),
                edgecolor='white', linewidth=0.8,
                label=f'με {segment}' if segment != EXTERNAL else segment)
        for y, (value, start) in enumerate(zip(values, left)):
            # Μόνο τα τμήματα που χωρούν αριθμό μέσα τους.
            if value >= 7:
                ax.text(start + value / 2, y, f'{value:.1f}%', ha='center',
                        va='center', fontsize=9, color='white', weight='bold')
        left += values

    # Το κρίσιμο νούμερο έξω από τη ράβδο, γιατί είναι πολύ μικρό για να χωρέσει μέσα.
    if EXTERNAL in wide.columns:
        for y, name in enumerate(rows):
            value = wide.loc[name, EXTERNAL]
            if value < 7:   # αλλιώς ο ίδιος αριθμός γράφεται ήδη μέσα στη ράβδο
                ax.annotate(f'εκτός: {value:.1f}%', xy=(100, y), xytext=(6, 0),
                            textcoords='offset points', va='center',
                            fontsize=9, color='0.25')

    ax.set_xlim(0, 100)
    ax.set_xlabel('Ποσοστό των ροών του υποδικτύου (%)')
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title('Πού πάει η κίνηση κάθε ιδιωτικού υποδικτύου\n'
                 'ροές και των οκτώ ημερών, κάθε ράβδος αθροίζει στο 100%',
                 fontsize=13)
    ax.legend(bbox_to_anchor=(1.14, 1), loc='upper left', frameon=False)
    return save(fig, 'subnets_peers.png', graphs_dir)


def plot_hourly(graphs_dir):
    """
    Η κατανομή της κίνησης μέσα στο 24ωρο, μία καμπύλη ανά υποδίκτυο.

    Το σχήμα είναι το επιχείρημα: καμπάνα στις εργάσιμες ώρες σημαίνει
    άνθρωποι μπροστά σε υπολογιστές, ευθεία πάνω στη γραμμή της ομοιόμορφης
    κατανομής σημαίνει μηχανή που δουλεύει με χρονοδιάγραμμα. Οι δύο
    συναθροιστικοί δείκτες (Business_Ratio, Hour_Entropy) λένε το ίδιο με
    έναν αριθμό, αλλά ένας αριθμός δεν ξεχωρίζει την ευθεία από τις ριπές.
    """
    views = (('all', 'Όλες τις ημέρες'), ('weekdays', 'Καθημερινές'))
    paths = {key: os.path.join(output_dir(), f'subnet_hourly_{key}.csv')
             for key, _ in views}
    if not all(os.path.exists(p) for p in paths.values()):
        return False

    colors = subnet_colors()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (key, caption) in zip(axes, views):
        table = pd.read_csv(paths[key], encoding='utf-8-sig', index_col=0)
        if table.empty:
            plt.close(fig)
            return False
        hours = [int(c) for c in table.columns]

        ax.axvspan(8, 17, color='0.92', zorder=0, label='Ωράριο 08:00–17:59')
        ax.axhline(UNIFORM_SHARE, ls='--', lw=1, color='0.45', zorder=1)
        for name in present(subnet_order(), table.index):
            ax.plot(hours, table.loc[name].to_numpy(dtype=float) * 100,
                    marker='o', ms=3, lw=1.8, color=colors.get(name),
                    label=name, zorder=2)
        ax.set_title(caption, fontsize=11)
        ax.set_xlabel('Ώρα της ημέρας')
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlim(-0.5, 23.5)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Ποσοστό ροών ενός τυπικού host (%)')
    axes[0].annotate(f'ομοιόμορφη κατανομή ({UNIFORM_SHARE:.2f}%)',
                     xy=(0.3, UNIFORM_SHARE), xytext=(0, 4),
                     textcoords='offset points', fontsize=8, color='0.35')

    fig.suptitle('Ωριαίο προφίλ ανά ιδιωτικό υποδίκτυο\n'
                 'κανονικοποίηση ανά host, χωρίς τις μερικές ημέρες και χωρίς '
                 f'hosts με λιγότερες από {SUBNET_MIN_FLOWS_HOURLY} ροές',
                 fontsize=13, y=1.06)
    axes[1].legend(title='Υποδίκτυο', bbox_to_anchor=(1.02, 1), loc='upper left')
    return save(fig, 'subnets_hourly.png', graphs_dir)


def plot_k(graphs_dir):
    """
    Τρεις όψεις του ίδιου ερωτήματος: πόσο εξαρτώνται τα συμπεράσματα από το k.

    Αριστερά το silhouette κάθε υποδικτύου χωριστά, με γεμάτο δείκτη στο k που
    επιλέχθηκε. Στο μέσο το ARI μεταξύ ετικέτας και υποδικτύου: όσο μένει
    κοντά στο μηδέν, τόσο λιγότερο ακολουθεί η ομαδοποίηση τα σύνορα των
    διευθύνσεων. Δεξιά η συμφωνία κοινής και χωριστής ομαδοποίησης ανά
    υποδίκτυο, που είναι μονότονη και σε αντίθετες κατευθύνσεις: όσο πιο
    πολλές ομάδες, τόσο καλύτερα βλέπει το μεγάλο υποδίκτυο και τόσο πιο πολύ
    διαμελίζει το μικρό. Με ένα μόνο k κανένα από τα τρία δεν φαίνεται.
    """
    files = {name: os.path.join(output_dir(), f'subnet_{name}.csv')
             for name in ('ari', 'agreement')}
    if not all(os.path.exists(p) for p in files.values()):
        return False

    ari = pd.read_csv(files['ari'], encoding='utf-8-sig')
    agree = pd.read_csv(files['agreement'], encoding='utf-8-sig')
    if ari.empty:
        return False

    # Οι δείκτες του kselect, ένα αρχείο ανά πληθυσμό. Τους διαβάζω από εκεί
    # και δεν τους ξαναϋπολογίζω: είναι οι ίδιοι αριθμοί που είδε ο χρήστης
    # όταν διάλεγε το k, και φέρνουν μαζί και την τυπική απόκλιση από το
    # bootstrap, που ένας σκέτος υπολογισμός silhouette δεν έχει.
    curves = {}
    for key, title, _ in subnet_populations():
        path = os.path.join(BASE_DIR, 'results', 'k_selection', f'k_selection_{key}.csv')
        if os.path.exists(path):
            curves[title.replace('Υποδίκτυο ', '')] = pd.read_csv(path).sort_values('k')

    colors = subnet_colors()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # Το k που τελικά χρησιμοποιήθηκε: όσες ομάδες βγήκαν στη χωριστή
    # ομαδοποίηση, όπως τις μέτρησε η σύγκριση.
    used = (agree.drop_duplicates('Υποδίκτυο').set_index('Υποδίκτυο')['Ομάδες_χωριστά']
            if not agree.empty else pd.Series(dtype=int))
    for name in present(subnet_order(), curves):
        part = curves[name]
        axes[0].plot(part['k'], part['Silhouette'], marker='o', ms=4, lw=1.8,
                     color=colors.get(name), label=name)
        if 'Sil_sd' in part.columns:
            axes[0].fill_between(part['k'], part['Silhouette'] - part['Sil_sd'],
                                 part['Silhouette'] + part['Sil_sd'],
                                 color=colors.get(name), alpha=0.15, lw=0)
        k_used = used.get(name)
        if k_used is not None:
            spot = part[part['k'] == int(k_used)]
            axes[0].scatter(spot['k'], spot['Silhouette'], s=130, zorder=3,
                            facecolor=colors.get(name), edgecolor='black',
                            linewidth=1.2)
    axes[0].set_title('Silhouette ανά πληθυσμό, χωριστά\n'
                      '(γεμάτος δείκτης: το k που χρησιμοποιήθηκε)', fontsize=11)
    axes[0].set_ylabel('Silhouette ± τυπική απόκλιση')

    axes[1].axhline(0, ls='-', lw=1, color='0.3', zorder=1)
    axes[1].plot(ari['k'], ari['ARI_με_υποδίκτυο'], marker='o', ms=5, lw=2,
                 color='firebrick', zorder=2)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_yticks(np.arange(-1, 1.01, 0.25))
    axes[1].annotate('1,0 = η ομαδοποίηση αναπαράγει\nακριβώς τα υποδίκτυα',
                     xy=(0.03, 0.97), xycoords='axes fraction', va='top',
                     fontsize=8, color='0.3')
    axes[1].annotate('0 = καμία σχέση με τη διεύθυνση',
                     xy=(0.03, 0.06), xycoords='axes fraction',
                     fontsize=8, color='0.3')
    axes[1].set_title('Κοινή ομαδοποίηση:\nARI ετικέτας με το υποδίκτυο', fontsize=11)
    axes[1].set_ylabel('ARI')

    if not agree.empty:
        for name in present(subnet_order(), agree['Υποδίκτυο']):
            part = agree[agree['Υποδίκτυο'] == name].sort_values('k_κοινής')
            axes[2].plot(part['k_κοινής'], part['ARI_κοινή_vs_χωριστή'],
                         marker='o', ms=4, lw=1.8, color=colors.get(name),
                         label=name)
        axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title('Συμφωνία κοινής και χωριστής\nομαδοποίησης, ανά υποδίκτυο',
                      fontsize=11)
    axes[2].set_ylabel('ARI')

    for ax in axes:
        ax.set_xlabel('k της ομαδοποίησης')
        ax.grid(linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0].legend(title='Υποδίκτυο', fontsize=8, title_fontsize=9)
    axes[2].legend(title='Υποδίκτυο', fontsize=8, title_fontsize=9)

    fig.suptitle('Πόσο εξαρτώνται τα συμπεράσματα από τον αριθμό των ομάδων',
                 fontsize=13, y=1.04)
    return save(fig, 'subnets_k.png', graphs_dir)


def run(graphs_dir=None):
    """Όλα τα γραφήματα των υποδικτύων. Γυρίζει True αν σχεδιάστηκε έστω ένα."""
    from src.config import BASE_DIR
    if graphs_dir is None:
        graphs_dir = os.path.join(BASE_DIR, 'results', 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)

    drawn = False
    for label, function in (('επαφές', plot_peers), ('ωριαίο προφίλ', plot_hourly),
                            ('εξάρτηση από το k', plot_k)):
        if function(graphs_dir):
            drawn = True
        else:
            print(f" -> Παραλείπεται το γράφημα «{label}»: λείπουν τα CSV του "
                  f"(τρέξε  python main.py --step subnets).")
    return drawn


if __name__ == "__main__":
    run()
