import os
import sys
import glob
import time
import argparse
import importlib
import traceback
from datetime import datetime

# Ταξινομημένα στάδια: key -> (τίτλος, module, προαπαιτούμενα, υπόδειξη αν λείπουν)
# Προαπαιτούμενα: λίστα από (κλειδί PATHS ή 'graphs', pattern). Αρκεί ΕΝΑ από αυτά να έχει αρχεία.
# Τα modules φορτώνονται μόνο όταν χρειάζονται, ώστε το --help να δουλεύει ακόμα κι αν λείπει το .env.
STEPS = {
    'clean': (
        'Καθαρισμός & Ανωνυμοποίηση (Raw -> Clean)', 'src.preprocessing.clean_anonymize',
        [('raw_data', '*.csv')],
        "Βάλε τα raw CSV στον φάκελο data/1_raw."),
    'aggregate': (
        'Αμφίδρομη Συνένωση Ροών (Clean -> Bidir)', 'src.preprocessing.aggregate_flows',
        [('processed_data', 'clean_*.csv')],
        "Τρέξε πρώτα:  python main.py --step clean"),
    'verify': (
        'Έλεγχος Ακεραιότητας Δεδομένων (QA)', 'src.utils.integrity',
        [('bidirectional_data', 'bidir_*.csv')],
        "Τρέξε πρώτα:  python main.py --step aggregate"),
    'features': (
        'Εξαγωγή Χαρακτηριστικών (Bidir -> Profiles)', 'src.features.extract_features',
        [('bidirectional_data', 'bidir_*.csv')],
        "Τρέξε πρώτα:  python main.py --step aggregate"),
    'privatescan': (
        'Εντοπισμός Ιδιωτικών Διευθύνσεων (RFC1918)', 'src.utils.private_scan',
        [('features', 'features_*.csv')],
        "Τρέξε πρώτα:  python main.py --step features"),
    'kselect': (
        'Επιλογή Αριθμού Clusters k (Δείκτες Εγκυρότητας)', 'src.models.select_k',
        [('features', 'features_*.csv')],
        "Τρέξε πρώτα:  python main.py --step features"),
    'sensitivity': (
        'Ευαισθησία στο Ποσοστό Ανωμαλιών', 'src.models.sensitivity',
        [('features', 'features_*.csv')],
        "Τρέξε πρώτα:  python main.py --step features"),
    'macro': (
        'Μακροσκοπική Ανάλυση (Εξωτερικό Δίκτυο)', 'src.models.macro_clustering',
        [('features', 'features_*.csv')],
        "Τρέξε πρώτα:  python main.py --step features"),
    'agreement': (
        'Συμφωνία Isolation Forest & DBSCAN (Πίνακας Συνάφειας)', 'src.models.method_agreement',
        [('features', 'features_*.csv')],
        "Τρέξε πρώτα:  python main.py --step features"),
    'micro': (
        'Μικροσκοπική Ανάλυση (Εσωτερικό Δίκτυο)', 'src.models.micro_profiling',
        [('features', 'features_*.csv')],
        "Τρέξε πρώτα:  python main.py --step features"),
    'stability': (
        'Ευστάθεια Ομάδων ανά Ημέρα', 'src.models.daily_stability',
        [('bidirectional_data', 'bidir_*.csv')],
        "Τρέξε πρώτα:  python main.py --step aggregate"),
    'peers': (
        'Επανεμφάνιση Peers (σάρωση ή P2P;)', 'src.models.peer_recurrence',
        [('bidirectional_data', 'bidir_*.csv')],
        "Τρέξε πρώτα:  python main.py --step aggregate"),
    'portcheck': (
        'Έλεγχος Σημασίας του Service_Port', 'src.models.port_semantics',
        [('processed_data', 'clean_*.csv')],
        "Τρέξε πρώτα:  python main.py --step clean"),
    'subnets': (
        'Ανάλυση Ιδιωτικών Υποδικτύων (χωριστά)', 'src.models.subnet_analysis',
        [('features', 'features_*.csv')],
        "Τρέξε πρώτα:  python main.py --step features"),
    'ablation': (
        'Ablation Χαρακτηριστικών (ποια οικογένεια κρατά τη δομή)',
        'src.models.ablation',
        [('results_macro', 'clean_macro_*.csv'), ('results_micro', 'micro_*.csv')],
        "Τρέξε πρώτα:  python main.py --step macro micro"),
    'hourly': (
        'Ωριαίο Προφίλ Δραστηριότητας ανά Ομάδα', 'src.models.hourly_profile',
        [('results_macro', 'clean_macro_*.csv'), ('results_micro', 'micro_*.csv')],
        "Τρέξε πρώτα:  python main.py --step macro micro"),
    'visualize': (
        'Οπτικοποιήσεις UMAP (Graphs)', 'src.visualizations.plot_results',
        [('results_macro', 'clean_macro_*.csv'), ('results_micro', 'micro_*.csv')],
        "Τρέξε πρώτα:  python main.py --step macro micro"),
}

# Στάδια που δέχονται παραμέτρους από τη γραμμή εντολών: {στάδιο: {όρισμα: επιλογή}}
STEP_OPTIONS = {
    'macro': {'n_clusters': 'k_macro'},
    'micro': {'n_clusters': 'k_micro'},
    'stability': {'k_macro': 'k_macro', 'k_micro': 'k_micro'},
}

# Στάδια που ΔΕΝ μπαίνουν στη συντόμευση [M]: η προετοιμασία των δεδομένων,
# που τρέχει μία φορά και δεν αλλάζει, και η επιλογή του k, που είναι ακριβή
# και τους δείκτες της τους δείχνουν ήδη τα macro/micro όταν ρωτούν για το k.
#
# Έξω μένει και ο έλεγχος του Service_Port: διαβάζει τα 4,3 GB των
# καθαρισμένων αρχείων και η απάντησή του αλλάζει μόνο αν αλλάξουν τα
# δεδομένα, όχι αν ξανατρέξουν τα μοντέλα. Παραμένει στο μενού ως στάδιο.
#
# Ορίζω τι ΜΕΝΕΙ ΕΞΩ και όχι τι μπαίνει, ώστε κάθε νέο στάδιο να προστίθεται
# αυτόματα στο [M]. Με γραμμένη λίστα, δύο στάδια (agreement, peers) έμειναν
# σιωπηλά έξω και φάνηκε μόνο όταν κάποιος πρόσεξε ότι λείπει μια φάση.
# Το 'privatescan' μένει έξω για τον ίδιο λόγο με το 'features': δεν είναι
# μοντέλο, είναι παραγωγή εισόδου. Ο κατάλογός του εξαρτάται μόνο από τα
# προφίλ και το κλειδί, άρα δεν αλλάζει όταν ξανατρέχουν τα μοντέλα — και
# κοστίζει 68 δευτερόλεπτα, που δεν έχει νόημα να πληρώνονται κάθε φορά.
M_EXCLUDED = ('clean', 'aggregate', 'verify', 'features', 'privatescan',
              'kselect', 'portcheck')

_CFG = {}


def model_steps():
    """Τα στάδια της συντόμευσης [M], με τη σειρά του pipeline."""
    return [key for key in STEPS if key not in M_EXCLUDED]


def load_config():
    """Φορτώνει το config την πρώτη φορά, με καθαρό μήνυμα αντί για traceback αν κάτι λείπει."""
    if not _CFG:
        try:
            from src import config
        except ValueError as e:
            print(f"\n[!] {e}\n    Δημιούργησε αρχείο .env στη ρίζα του project, με μία γραμμή:"
                  f"\n        THESIS_SALT_KEY=<τυχαίο κλειδί>"
                  f"\n    Φτιάξε κλειδί με:  py -c \"import secrets;print(secrets.token_hex(32))\"")
            sys.exit(1)
        except ImportError as e:
            print(f"\n[!] Λείπει βιβλιοθήκη: {e}\n    Δοκίμασε:  pip install -r requirements.txt")
            sys.exit(1)
        _CFG['PATHS'] = config.PATHS
        _CFG['BASE_DIR'] = config.BASE_DIR
    return _CFG['PATHS'], _CFG['BASE_DIR']


def graphs_dir():
    _, base = load_config()
    return os.path.join(base, 'results', 'graphs')


def kselect_dir():
    _, base = load_config()
    return os.path.join(base, 'results', 'k_selection')


def stability_dir():
    _, base = load_config()
    return os.path.join(base, 'results', 'stability')


def sensitivity_dir():
    _, base = load_config()
    return os.path.join(base, 'results', 'sensitivity')


def peers_dir():
    _, base = load_config()
    return os.path.join(base, 'results', 'peers')


def hourly_dir():
    _, base = load_config()
    return os.path.join(base, 'results', 'hourly')


def ports_dir():
    _, base = load_config()
    return os.path.join(base, 'results', 'ports')


def subnets_dir():
    _, base = load_config()
    return os.path.join(base, 'results', 'subnets')


def count_files(folder, pattern="*.csv"):
    if not os.path.exists(folder):
        return 0
    return len(glob.glob(os.path.join(folder, pattern)))


def folder_of(key):
    paths, _ = load_config()
    return graphs_dir() if key == 'graphs' else paths[key]


def missing_prerequisite(key):
    """Επιστρέφει την υπόδειξη αν λείπουν τα δεδομένα εισόδου του σταδίου, αλλιώς None."""
    _, _, requirements, hint = STEPS[key]
    if any(count_files(folder_of(k), pattern) for k, pattern in requirements):
        return None
    return hint


def log_failure(key):
    """Γράφει το πλήρες traceback σε results/pipeline.log (να καλείται μέσα σε except)."""
    _, base = load_config()
    results_dir = os.path.join(base, 'results')
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, 'pipeline.log')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] Αποτυχία σταδίου '{key}'\n{traceback.format_exc()}\n")
    return log_path


def print_status():
    """Εμφανίζει μια γρήγορη σύνοψη των αρχείων που υπάρχουν ήδη στους φακέλους."""
    paths, _ = load_config()
    macro_clean = count_files(paths['results_macro'], 'clean_macro_*.csv')
    macro_anom = count_files(paths['results_macro'], 'anomalies_macro_*.csv')
    print("\n" + "=" * 60)
    print("           ΚΑΤΑΣΤΑΣΗ ΑΡΧΕΙΩΝ PIPELINE (STATUS)")
    print("=" * 60)
    print(f"  [1] Raw Data (1_raw):               {count_files(paths['raw_data'])} CSV")
    print(f"  [2] Processed (2_processed):        {count_files(paths['processed_data'], 'clean_*.csv')} CSV")
    print(f"  [3] Bidirectional (3_bidirectional):{count_files(paths['bidirectional_data'], 'bidir_*.csv')} CSV")
    print(f"  [4] Profiles (4_profiles):          {count_files(paths['features'], 'features_*.csv')} CSV")
    print(f"  [5] Δείκτες k (k_selection):        {count_files(kselect_dir(), 'k_selection_*.csv')} CSV")
    print(f"  [6] Ευαισθησία ανωμαλιών:           {count_files(sensitivity_dir(), '*.csv')} CSV")
    print(f"  [7] Macro Results (clean/anomalies):{macro_clean} / {macro_anom} CSV")
    print(f"  [8] Συμφωνία Iso/DBSCAN:            {count_files(paths['results_macro'], 'contingency_*.csv')} CSV")
    print(f"  [9] Micro Results:                  {count_files(paths['results_micro'], 'micro_*.csv')} CSV")
    print(f" [10] Ευστάθεια ανά ημέρα:            {count_files(stability_dir(), 'ari_*.csv')} CSV")
    print(f" [11] Επανεμφάνιση peers/θυρών:       {count_files(peers_dir(), '*.csv')} CSV")
    print(f" [12] Έλεγχος Service_Port:           {count_files(ports_dir(), '*.csv')} CSV")
    print(f" [13] Ανάλυση υποδικτύων:             {count_files(subnets_dir(), '*.csv')} CSV")
    print(f" [14] Ωριαίο προφίλ ανά ομάδα:        {count_files(hourly_dir(), '*.csv')} CSV")
    print(f" [15] Graphs (UMAP):                  {count_files(graphs_dir(), '*.png')} PNG")
    print("=" * 60)


def run_step(key, options=None):
    """Εκτελεί ένα στάδιο. Επιστρέφει (ok, μήνυμα). Δεν αφήνει εξαιρέσεις να ξεφύγουν."""
    hint = missing_prerequisite(key)
    if hint:
        return False, f"Δεν βρέθηκαν δεδομένα εισόδου. {hint}"
    try:
        module = importlib.import_module(STEPS[key][1])
        kwargs = {}
        for argument, option_name in STEP_OPTIONS.get(key, {}).items():
            if options and options.get(option_name) is not None:
                kwargs[argument] = options[option_name]
        module.run(**kwargs)
    except KeyboardInterrupt:
        return False, "Διακόπηκε από τον χρήστη (Ctrl-C)."
    except ImportError as e:
        log_path = log_failure(key)
        return False, (f"Λείπει βιβλιοθήκη ή module: {e}. Δοκίμασε: pip install -r requirements.txt "
                       f"(λεπτομέρειες στο {log_path})")
    except Exception as e:
        log_path = log_failure(key)
        return False, f"{type(e).__name__}: {e} (πλήρες traceback στο {log_path})"
    return True, ""


def execute_steps(step_keys, options=None):
    """
    Εκτελεί τα επιλεγμένα στάδια με τη σειρά του pipeline (χωρίς διπλότυπα), με χρονομέτρηση.
    Σταματά στο πρώτο στάδιο που αποτυγχάνει. Επιστρέφει True μόνο αν πέτυχαν όλα.
    """
    selected = set(step_keys)
    keys = [k for k in STEPS if k in selected]
    total_start = time.perf_counter()
    print(f"\n=== Έναρξη Εκτέλεσης: {', '.join(k.upper() for k in keys)} ===")

    summary = []
    all_ok = True
    for key in keys:
        number = list(STEPS).index(key) + 1
        print(f"\n>>> Φάση {number}/{len(STEPS)}: {STEPS[key][0]}")
        start = time.perf_counter()
        ok, message = run_step(key, options)
        elapsed = time.perf_counter() - start
        summary.append((key, ok, elapsed))
        if ok:
            print(f"   [Χρόνος σταδίου '{key}': {elapsed:.1f} δευτερόλεπτα]")
        else:
            print(f"\n[!] Το στάδιο '{key}' ΑΠΕΤΥΧΕ: {message}")
            all_ok = False
            break

    total_elapsed = time.perf_counter() - total_start
    mins, secs = divmod(total_elapsed, 60)
    print("\n" + "-" * 60)
    for key, ok, elapsed in summary:
        print(f"  {'[OK]       ' if ok else '[ΑΠΟΤΥΧΙΑ]'} {key:<10} {elapsed:8.1f} δ")
    not_run = [k for k in keys if k not in {s[0] for s in summary}]
    for key in not_run:
        print(f"  [ΜΗ ΕΚΤΕΛΕΣΘΗΚΕ] {key}")
    print("-" * 60)
    if all_ok:
        print(f"=== Ολοκληρώθηκε επιτυχώς σε {int(mins)}λ {secs:.1f}δ ===")
    else:
        print(f"=== Το pipeline ΔΙΑΚΟΠΗΚΕ μετά από {int(mins)}λ {secs:.1f}δ ===")

    write_manifest(keys, all_ok, total_elapsed, options)
    return all_ok


def write_manifest(keys, all_ok, total_elapsed, options):
    """
    Καταγραφή της εκτέλεσης σε results/run_manifest.json.

    Σε try/except επειδή το manifest είναι τεκμηρίωση, όχι αποτέλεσμα: αν
    αποτύχει δεν έχει νόημα να χαθεί μια εκτέλεση που κράτησε λεπτά.
    """
    try:
        from src.utils import manifest
        path = manifest.write(keys, all_ok, total_elapsed, options)
        print(f"  Καταγραφή εκτέλεσης: {path}")
    except Exception as e:
        print(f"  [!] Δεν γράφτηκε το run_manifest.json ({type(e).__name__}: {e})")


def clean_intermediate_files():
    """Διαγράφει τα παραγόμενα CSV/PNG (όχι τα raw data και όχι άλλα αρχεία) μετά από επιβεβαίωση."""
    paths, _ = load_config()
    folders = [
        paths['processed_data'],
        paths['bidirectional_data'],
        paths['features'],
        paths['results_macro'],
        paths['results_micro'],
        kselect_dir(),
        sensitivity_dir(),
        stability_dir(),
        peers_dir(),
        hourly_dir(),
        ports_dir(),
        subnets_dir(),
        graphs_dir(),
    ]
    targets = [f for folder in folders for pattern in ('*.csv', '*.png')
               for f in glob.glob(os.path.join(folder, pattern))]
    if not targets:
        print("\nΔεν υπάρχουν παραγόμενα αρχεία προς διαγραφή.")
        return

    confirm = ask(f"\nΠροσοχή! Θα διαγραφούν {len(targets)} παραγόμενα αρχεία (CSV/PNG) από τα data/2..4 και results/.\n"
                  "Τα Raw Data ΔΕΝ πειράζονται. Αν έχεις τελικά γραφήματα για τη διπλωματική,\n"
                  "αντίγραψέ τα πρώτα σε άλλον φάκελο.\n"
                  "Είσαι σίγουρος; (y/n): ")
    if not confirm or confirm.strip().lower() != 'y':
        print(" -> Ακύρωση διαγραφής.")
        return

    for f in targets:
        os.remove(f)
    print(f" -> Διαγράφηκαν επιτυχώς {len(targets)} παραγόμενα αρχεία.")


def ask(prompt):
    """input() που δεν σκάει όταν κλείσει το stdin (επιστρέφει None)."""
    try:
        return input(prompt)
    except EOFError:
        return None


def interactive_menu():
    """Διαδραστικό μενού όταν το script εκτελείται χωρίς παραμέτρους."""
    step_list = list(STEPS.keys())
    while True:
        print_status()
        print("\nΕΠΙΛΟΓΕΣ ΕΚΤΕΛΕΣΗΣ:")
        for idx, key in enumerate(step_list, start=1):
            print(f"  [{idx}] {STEPS[key][0]}")
        print("-" * 60)
        print("  [A] Εκτέλεση ΟΛΩΝ από την αρχή")
        print(f"  [M] Μόνο η ανάλυση: {len(model_steps())} στάδια "
              f"(μοντέλα, έλεγχοι, γραφήματα — χωρίς προετοιμασία και επιλογή k)")
        print("  [R] Εκκαθάριση παραγόμενων αρχείων (Reset Pipeline)")
        print("  [0] Έξοδος")
        print("-" * 60)

        choice = ask("Επίλεξε ενέργεια (π.χ. M, ή 6,7,8): ")
        if choice is None:
            break
        choice = choice.strip()

        if choice == '0' or choice.lower() in ['q', 'exit']:
            print("Έξοδος από το πρόγραμμα.")
            break
        elif choice.lower() == 'a':
            execute_steps(step_list)
        elif choice.lower() == 'm':
            execute_steps(model_steps())
        elif choice.lower() == 'r':
            clean_intermediate_files()
        else:
            # Υποστήριξη μεμονωμένων ή πολλαπλών αριθμών (π.χ. "4,5" ή "3")
            selected_keys = []
            valid = True
            for part in choice.replace(' ', ',').split(','):
                if not part:
                    continue
                if part.isdigit() and 1 <= int(part) <= len(step_list):
                    selected_keys.append(step_list[int(part) - 1])
                else:
                    valid = False
                    break

            if valid and selected_keys:
                execute_steps(selected_keys)  # ταξινομείται αυτόματα με τη σειρά του pipeline
            else:
                print("\n[!] Μη έγκυρη επιλογή. Δοκίμασε ξανά.")

        ask("\nΠάτησε Enter για επιστροφή στο μενού...")


def main():
    parser = argparse.ArgumentParser(
        description="NetFlow Analysis Pipeline",
        epilog="Παραδείγματα:\n"
               "  python main.py                            (διαδραστικό μενού)\n"
               "  python main.py --step all                 (όλα τα στάδια)\n"
               "  python main.py --step verify              (έλεγχος ακεραιότητας)\n"
               "  python main.py --step kselect             (μόνο οι δείκτες για το k)\n"
               "  python main.py --step macro --k-macro 4   (χωρίς ερώτηση)\n"
               "  python main.py --step features macro micro visualize\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--step',
        nargs='+',
        choices=['all'] + list(STEPS.keys()),
        help="Εκτέλεση ενός ή περισσότερων σταδίων απευθείας. Εκτελούνται πάντα με τη σειρά του pipeline."
    )
    parser.add_argument(
        '--k-macro', type=int, metavar='N',
        help="Αριθμός clusters για τη μακροσκοπική ανάλυση. Παρακάμπτει την ερώτηση."
    )
    parser.add_argument(
        '--k-micro', type=int, metavar='N',
        help="Αριθμός clusters για τη μικροσκοπική ανάλυση. Παρακάμπτει την ερώτηση."
    )

    args = parser.parse_args()
    for name, value in (('--k-macro', args.k_macro), ('--k-micro', args.k_micro)):
        if value is not None and value < 2:
            parser.error(f"Το {name} πρέπει να είναι >= 2.")
    options = {'k_macro': args.k_macro, 'k_micro': args.k_micro}

    # Χωρίς παραμέτρους: διαδραστικό μενού, αλλά μόνο αν υπάρχει τερματικό (όχι nohup/cron/redirect)
    if not args.step:
        if not sys.stdin.isatty():
            parser.print_help()
            sys.exit(2)
        load_config()
        try:
            interactive_menu()
        except KeyboardInterrupt:
            print("\nΈξοδος από το πρόγραμμα.")
        return

    load_config()
    keys = list(STEPS.keys()) if 'all' in args.step else args.step
    sys.exit(0 if execute_steps(keys, options) else 1)


if __name__ == "__main__":
    main()
