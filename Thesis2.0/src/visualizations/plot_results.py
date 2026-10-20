# src/visualizations/plot_results.py
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # χωρίς GUI, ώστε να τρέχει και σε server
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from src.config import PATHS, BASE_DIR, RANDOM_STATE
from src.utils.scaling import scale_features

# (όνομα, φάκελος, πρότυπο αρχείων). Το micro περιέχει τους εσωτερικούς hosts.
SOURCES = [
    ("Macro", PATHS['results_macro'], "clean_macro_*.csv"),
    ("Micro", PATHS['results_micro'], "micro_*.csv"),
]

# (στήλη ετικετών, τίτλος)
LABEL_COLUMNS = [
    ("Cluster", "K-Means"),
    ("DBSCAN_Cluster", "DBSCAN"),
]

# Η σύνθεση ανά ημέρα, όπως τη γράφει το στάδιο ευστάθειας μετά την αντιστοίχιση ετικετών
COMPOSITIONS = [
    ("macro", "Εξωτερικοί hosts"),
    ("micro", "Εσωτερικοί hosts"),
]

# Το ωριαίο προφίλ, όπως το γράφει το στάδιο hourly
HOURLY = [
    ("macro", "Εξωτερικοί hosts"),
    ("micro", "Εσωτερικοί hosts"),
]

# Η τιμή που θα είχε κάθε ώρα αν η κίνηση ήταν ομοιόμορφη στο 24ωρο.
# Χωρίς αυτή τη γραμμή, μια «επίπεδη» καμπύλη δεν διαβάζεται ως επίπεδη.
UNIFORM_SHARE = 100 / 24


def embed(X, seed=RANDOM_STATE):
    """
    UMAP πάνω στα ΜΟΝΑΔΙΚΑ προφίλ και χαρτογράφηση πίσω στους hosts.

    Με πάνω από τα μισά προφίλ ταυτόσημα ο γράφος γειτνίασης του UMAP
    εκφυλίζεται: η φασματική αρχικοποίηση αποτυγχάνει ("too small an
    eigengap"), πέφτει σε τυχαία και το γράφημα βγαίνει διάχυτο, χωρίς δομή.
    Στα μοναδικά σημεία δεν συμβαίνει αυτό, και τρέχει αρκετά πιο γρήγορα.

    Τα ταυτόσημα προφίλ παίρνουν τις ίδιες συντεταγμένες, οπότε προσθέτω λίγο
    θόρυβο ώστε οι πυκνές μάζες να φαίνονται ως μάζες και όχι ως ένα σημείο.
    Ο θόρυβος είναι καθαρά οπτικός, 0,5% του εύρους.
    """
    U, inverse = np.unique(np.round(X, 6), axis=0, return_inverse=True)
    print(f"    - {len(X)} hosts -> {len(U)} μοναδικά προφίλ ({100 * len(U) / len(X):.1f}%)")

    reducer = umap.UMAP(n_neighbors=min(15, len(U) - 1), min_dist=0.1, random_state=seed)
    embedding = reducer.fit_transform(U)[np.ravel(inverse)]

    if len(U) < len(X):
        rng = np.random.default_rng(seed)
        spread = 0.005 * (embedding.max(axis=0) - embedding.min(axis=0))
        embedding = embedding + rng.normal(0, spread, size=embedding.shape)
    return embedding


def marker_style(n_points):
    """
    Μέγεθος και διαφάνεια των σημείων, ανάλογα με το πλήθος τους.

    Στο macro σχεδιάζονται ~121.000 hosts. Με s=50 και μαύρο περίγραμμα το
    scatter γίνεται μια συμπαγής κηλίδα και δεν φαίνεται τίποτα.
    """
    if n_points > 20000:
        return dict(s=2, alpha=0.35, edgecolor='none')
    if n_points > 2000:
        return dict(s=10, alpha=0.5, edgecolor='none')
    return dict(s=50, alpha=0.7, edgecolor='k')


def plot_embedding(df, label_col, method_name, source_name, filename, out_img):
    """Scatter plot του UMAP embedding. Αν υπάρχουν >10 clusters, κρατάει τα 10 μεγαλύτερα στο legend."""
    raw_labels = df[label_col]
    
    # Εντοπισμός των 10 μεγαλύτερων clusters (εξαιρώντας το -1 = Noise)
    valid_counts = raw_labels[raw_labels != -1].value_counts()
    top_clusters = set(valid_counts.head(10).index)

    def format_label(l):
        if l == -1:
            return 'Noise'
        if l in top_clusters:
            return f'Cluster {l}'
        return 'Other Clusters'

    labels = raw_labels.map(format_label)
    cluster_names = sorted(
        [n for n in labels.unique() if n not in ('Noise', 'Other Clusters')],
        key=lambda x: int(x.split()[-1])
    )
    order = list(cluster_names)  # αντίγραφο: αλλιώς τα append παρακάτω αλλάζουν και το cluster_names
    if 'Other Clusters' in set(labels):
        order.append('Other Clusters')
    if 'Noise' in set(labels):
        order.append('Noise')

    colors = sns.color_palette('tab10', n_colors=max(len(cluster_names), 1))
    palette = {name: colors[i] for i, name in enumerate(cluster_names)}
    palette['Other Clusters'] = 'darkgray'
    palette['Noise'] = 'lightgray'

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=df['UMAP_X'], y=df['UMAP_Y'],
        hue=labels, hue_order=order, palette=palette,
        **marker_style(len(df))
    )
    # Στα πυκνά γραφήματα τα σημεία του legend μένουν μεγάλα και ορατά.
    legend = plt.gca().get_legend()
    if legend is not None:
        # legend_handles: matplotlib >= 3.7, legendHandles: παλιότερες εκδόσεις
        handles = getattr(legend, 'legend_handles', None) or getattr(legend, 'legendHandles', [])
        for handle in handles:
            # Ανάλογα με την έκδοση, το seaborn δίνει Line2D ή PathCollection
            if hasattr(handle, 'set_sizes'):
                handle.set_sizes([60])
            elif hasattr(handle, 'set_markersize'):
                handle.set_markersize(8)
            handle.set_alpha(1.0)
    plt.title(f'Οπτικοποίηση Κίνησης με UMAP ({method_name} Clusters)\n{filename}', fontsize=14)
    plt.xlabel('UMAP Διάσταση 1')
    plt.ylabel('UMAP Διάσταση 2')
    plt.legend(title=f'{source_name} {method_name}', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> [ΕΠΙΤΥΧΙΑ] Το γράφημα αποθηκεύτηκε στο: {out_img}")


def cluster_colors(labels):
    """Ίδια χρώματα με τα γραφήματα UMAP: tab10, με τη σειρά του αριθμού."""
    palette = sns.color_palette('tab10', n_colors=max(max(labels) + 1, 1))
    return [palette[c] for c in labels]


def plot_composition(level, title, graphs_dir):
    """
    Πόσοι hosts σε κάθε ομάδα, ανά ημέρα.

    Έχει νόημα μόνο επειδή οι ετικέτες έχουν αντιστοιχιστεί στη συνολική
    ομαδοποίηση από το στάδιο ευστάθειας. Με τις ακατέργαστες ετικέτες του K-Means το
    γράφημα θα ήταν παραπλανητικό: η ομάδα 1 της Τρίτης δεν είναι η ομάδα 1
    της Τετάρτης.

    Δύο όψεις: αριστερά τα απόλυτα πλήθη, που δείχνουν και πόσοι hosts
    εμφανίζονται κάθε ημέρα, δεξιά τα ποσοστά, που δείχνουν αν αλλάζει η
    σύνθεση ανεξάρτητα από το μέγεθος.
    """
    path = os.path.join(BASE_DIR, 'results', 'stability', f'composition_{level}.csv')
    if not os.path.exists(path):
        return False

    table = pd.read_csv(path, encoding='utf-8-sig').set_index('Ημέρα')
    if table.empty:
        return False

    labels = [int(c.split('_')[-1]) for c in table.columns]
    colors = cluster_colors(labels)
    days = [f"{str(d)[6:8]}/{str(d)[4:6]}" for d in table.index]
    shares = table.div(table.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, data, ylabel in ((axes[0], table, 'Hosts'),
                             (axes[1], shares, 'Ποσοστό (%)')):
        bottom = np.zeros(len(data))
        for column, color, label in zip(data.columns, colors, labels):
            values = data[column].to_numpy(dtype=float)
            ax.bar(days, values, bottom=bottom, color=color,
                   label=f'Cluster {label}', edgecolor='white', linewidth=0.6)
            bottom += values
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Ημέρα')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
    axes[1].set_ylim(0, 100)

    fig.suptitle(f'Σύνθεση ομάδων ανά ημέρα — {title}\n'
                 f'ετικέτες αντιστοιχισμένες στη συνολική ομαδοποίηση',
                 fontsize=13)
    axes[1].legend(title='Ομάδα', bbox_to_anchor=(1.02, 1), loc='upper left')
    out_img = os.path.join(graphs_dir, f'composition_{level}.png')
    fig.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" -> [ΕΠΙΤΥΧΙΑ] Το γράφημα αποθηκεύτηκε στο: {out_img}")
    return True


def plot_hourly(level, title, graphs_dir):
    """
    Η κατανομή της κίνησης μέσα στο 24ωρο, μία καμπύλη ανά ομάδα.

    Αυτό το γράφημα υπάρχει για να μη στηρίζεται η απόδοση ρόλων σε δύο
    συναθροιστικούς δείκτες. Το Business_Ratio λέει πόση κίνηση έπεσε μέσα στο
    ωράριο· εδώ φαίνεται το σχήμα, που είναι το πραγματικό αποτύπωμα: καμπάνα
    στις εργάσιμες ώρες για σταθμό εργασίας, ευθεία για συσκευή που δουλεύει
    συνεχώς.

    Δύο όψεις: αριστερά όλες οι ημέρες, δεξιά μόνο οι καθημερινές, όπου το
    σήμα του ωραρίου είναι καθαρότερο. Η διακεκομμένη γραμμή στο 4,17% είναι η
    ομοιόμορφη κατανομή (1/24), δηλαδή η τιμή αναφοράς: πάνω της σημαίνει
    συγκέντρωση, πάνω της παντού σημαίνει τίποτα.
    """
    folder = os.path.join(BASE_DIR, 'results', 'hourly')
    views = (('all', 'Όλες τις ημέρες'), ('weekdays', 'Καθημερινές'))
    paths = {key: os.path.join(folder, f'hourly_{level}_{key}.csv') for key, _ in views}
    if not all(os.path.exists(p) for p in paths.values()):
        return False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (key, caption) in zip(axes, views):
        table = pd.read_csv(paths[key], encoding='utf-8-sig').set_index('Ομάδα')
        if table.empty:
            plt.close(fig)
            return False
        hours = [int(c) for c in table.columns]
        labels = [int(i) for i in table.index]

        # Η ζώνη του ωραρίου σχεδιάζεται πρώτη και από κάτω, ώστε να μην
        # σκεπάζει τις καμπύλες.
        ax.axvspan(8, 17, color='0.92', zorder=0, label='Ωράριο 08:00–17:59')
        ax.axhline(UNIFORM_SHARE, ls='--', lw=1, color='0.45', zorder=1)
        for (_, row), color, label in zip(table.iterrows(),
                                          cluster_colors(labels), labels):
            ax.plot(hours, row.to_numpy(dtype=float) * 100, marker='o', ms=3,
                    lw=1.8, color=color, label=f'Cluster {label}', zorder=2)
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

    # y=1.06: με δύο γραμμές τίτλου και τίτλους ανά υπογράφημα, ο προεπιλεγμένος
    # suptitle πέφτει πάνω τους. Το bbox_inches='tight' μεγαλώνει τον καμβά.
    fig.suptitle(f'Ωριαίο προφίλ δραστηριότητας ανά ομάδα — {title}\n'
                 f'κανονικοποίηση ανά host, χωρίς τις μερικές ημέρες',
                 fontsize=13, y=1.06)
    axes[1].legend(title='Ομάδα', bbox_to_anchor=(1.02, 1), loc='upper left')
    out_img = os.path.join(graphs_dir, f'hourly_{level}.png')
    fig.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" -> [ΕΠΙΤΥΧΙΑ] Το γράφημα αποθηκεύτηκε στο: {out_img}")
    return True


def run():
    print("\n--- Έναρξη: Οπτικοποιήσεις (Data Visualization) ---")

    # Δημιουργία φακέλου για τα γραφικά
    graphs_dir = os.path.join(BASE_DIR, 'results', 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)

    found_any = False

    for source_name, folder, pattern in SOURCES:
        for file_path in glob.glob(os.path.join(folder, pattern)):
            found_any = True
            filename = os.path.basename(file_path)
            print(f"\nΔημιουργία γραφημάτων για: {filename}")

            df = pd.read_csv(file_path)

            # Αν το αρχείο είναι πολύ μικρό, το παραλείπουμε
            if len(df) < 10:
                print(" -> Πολύ λίγα δεδομένα για UMAP. Παράλειψη.")
                continue

            # Το UMAP τρέχει στα ίδια κλιμακωμένα χαρακτηριστικά με τα μοντέλα
            # (log1p + StandardScaler). Χωρίς scaling την εικόνα θα την καθόριζαν
            # σχεδόν μόνο τα bytes.
            X = scale_features(df)

            print(" -> Εκτέλεση αλγορίθμου UMAP (Μείωση Διαστάσεων)...")
            embedding = embed(X)
            df['UMAP_X'] = embedding[:, 0]
            df['UMAP_Y'] = embedding[:, 1]

            print(" -> Σχεδιασμός Scatter Plots...")
            for label_col, method_name in LABEL_COLUMNS:
                if label_col not in df.columns:
                    continue
                out_img = os.path.join(
                    graphs_dir, filename.replace('.csv', f'_UMAP_{method_name.replace("-", "")}.png'))
                plot_embedding(df, label_col, method_name, source_name, filename, out_img)

    print("\nΣύνθεση ομάδων ανά ημέρα:")
    for level, title in COMPOSITIONS:
        if plot_composition(level, title, graphs_dir):
            found_any = True
        else:
            print(f" -> Δεν βρέθηκε composition_{level}.csv (τρέξε το στάδιο stability).")

    print("\nΩριαίο προφίλ ανά ομάδα:")
    for level, title in HOURLY:
        if plot_hourly(level, title, graphs_dir):
            found_any = True
        else:
            print(f" -> Δεν βρέθηκαν τα hourly_{level}_*.csv (τρέξε το στάδιο hourly).")

    if not found_any:
        print("Προειδοποίηση: Δεν βρέθηκαν αρχεία για οπτικοποίηση.")
        return

    print("\n--- Ολοκληρώθηκαν οι Οπτικοποιήσεις ---")


if __name__ == "__main__":
    run()
