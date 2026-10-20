import os
from dotenv import load_dotenv

# 1. Φόρτωση Μεταβλητών Περιβάλλοντος (.env)
load_dotenv()

SALT_KEY_STR = os.getenv("THESIS_SALT_KEY")
if not SALT_KEY_STR:
    raise ValueError("ΣΦΑΛΜΑ: Το THESIS_SALT_KEY δεν βρέθηκε στο .env αρχείο.")
SALT_KEY_BYTES = SALT_KEY_STR.encode('utf-8')

# 2. Ρυθμίσεις Δικτύου
INTERNAL_SUBNET = "192.168.7." #! ΕΔΩ ΚΡΙΒΕΤΑΙ ΕΝΑ ΑΠΟ ΤΑ ΛΑΘΗ!! #

# 3. Ρυθμίσεις Μηχανικής Μάθησης
CONTAMINATION_RATE = 0.015
MACRO_CLUSTERS = 3
MICRO_CLUSTERS = 5
RANDOM_STATE = 42

# --- DBSCAN ---------------------------------------------------------------
# Στο macro πάνω από τα μισά προφίλ είναι ταυτόσημα μεταξύ τους (52,5% με τα
# 15 χαρακτηριστικά, 69% όταν ήταν 9): επαφές με μία ροή, ίδια bytes, ίδια θύρα. Αυτό χάλασε τον DBSCAN με δύο τρόπους:
#   α) το k-distance heuristic έβγαζε eps ~ 0, επειδή οι μισοί hosts έχουν
#      δίδυμα σε απόσταση 0,
#   β) ο eps-γράφος έφτανε τα 10^9 ακμές και τελείωνε σε MemoryError.
# Το λύνω με dedup + sample_weight (ισοδύναμο με DBSCAN στο πλήρες σύνολο)
# και με min_samples ανάλογο του μεγέθους. Λεπτομέρειες στο dbscan_clustering.py.

# Με DBSCAN_EPS = None το eps βγαίνει αυτόματα από το k-distance graph: το
# ποσοστημόριο DBSCAN_EPS_PERCENTILE των αποστάσεων προς τον k-οστό γείτονα,
# μετρημένο στα μοναδικά προφίλ, με k = min_samples.
DBSCAN_EPS = None
DBSCAN_EPS_PERCENTILE = 50

# min_samples = clip(DBSCAN_MIN_SAMPLES_FRACTION * n_hosts,
#                    DBSCAN_MIN_SAMPLES, DBSCAN_MIN_SAMPLES_MAX)
# Στο macro (121k hosts) βγαίνει 182, στο micro (64 hosts) 5. Με σταθερό
# min_samples=5 το macro έβγαζε ~660 ομάδες που δεν σήμαιναν τίποτα.
DBSCAN_MIN_SAMPLES = 5
DBSCAN_MIN_SAMPLES_FRACTION = 0.0015
DBSCAN_MIN_SAMPLES_MAX = 500

# Ανώτατο πλήθος ακμών του eps-γράφου. Αν η εκτίμηση το ξεπεράσει, το eps
# συρρικνώνεται σταδιακά: προτιμώ φραγμή μνήμης από MemoryError.
DBSCAN_MAX_EDGES = 40_000_000

# 4. Χαρακτηριστικά των μοντέλων.
# LOG_FEATURES: μεγέθη με βαριά ουρά -> log1p και μετά StandardScaler
# RATIO_FEATURES: λόγοι στο [0, 1] -> μόνο StandardScaler
#
# Τα δύο τελευταία πεδία του είναι In Byte και Out Byte, και το Out Byte είναι 0 σε όλες τις 25,4 εκατ.
# εγγραφές. Τα έβγαλα από το μοντέλο επειδή είναι σχεδόν διπλότυπα άλλων 
# (στο CSV εξάγονται κανονικά, τα κρατώ για την ερμηνεία):
# Οι δύο πρώτοι συντελεστές είναι πάνω στις log1p τιμές, δηλαδή στην κλίμακα
# που βλέπει ο K-Means. Στις ακατέργαστες βγαίνουν 0,11 και 0,14: τη διασπορά
# την κρατούν λίγες ακραίες τιμές και ο Pearson δεν λέει τίποτα εκεί. Το
# σημειώνω ώστε να μη φανεί αντίφαση αν τους ξαναυπολογίσει κανείς στο CSV.
#   Unique_Ports   -> 0,90 με Port_Entropy (κρατώ το δεύτερο, σταθμίζει με
#                     τα πλήθη ροών αντί να μετράει μόνο θύρες)
#   Flows_per_Peer -> 0,98 με Total_Flows
#   UDP_Ratio      -> -0,97 με TCP_Ratio (λόγος στο [0,1], χωρίς log· μόλις
#                     1,3% των hosts έχει καθόλου κίνηση εκτός TCP/UDP και
#                     μόλις 0,16% των ροών, οπότε είναι συμπληρωματικά)
# Ένα διπλότυπο χαρακτηριστικό μετράει δύο φορές στην απόσταση του K-Means
# και βαραίνει σιωπηλά μία διάσταση έναντι των υπολοίπων.
LOG_FEATURES = ['Bytes_In', 'Bytes_Out', 'Total_Flows',
                'Unique_Peers', 'Avg_Bytes_Per_Flow']
# LINEAR_FEATURES: μικρά φραγμένα πλήθη -> μόνο StandardScaler (χωρίς log1p,
# δεν έχουν βαριά ουρά: Active_Hours στο [0,24], Active_Days στο [0,8]).
LINEAR_FEATURES = ['Active_Hours', 'Active_Days', 'Port_Entropy']
RATIO_FEATURES = ['Out_Ratio', 'TCP_Ratio',
                  'Hour_Entropy', 'Business_Ratio', 'Weekend_Ratio',
                  'Well_Known_Ratio', 'P2P_Port_Ratio']
FEATURE_COLUMNS = LOG_FEATURES + LINEAR_FEATURES + RATIO_FEATURES

# --- Χαρακτηριστικά θυρών -------------------------------------------------
# Θύρες κάτω από 1024: καταχωρημένες υπηρεσίες (SSH 22, HTTP 80, HTTPS 443,
# DNS 53...). Ένας host που μιλά κυρίως εκεί κάνει "κανονική" χρήση υπηρεσιών. #! Το άλλο λάθος. Μέσα σε αυτά τα τις θύρες βρέθηκε port scanning #
WELL_KNOWN_PORT_MAX = 1024

# Τεκμηριωμένες θύρες P2P. Δεν καλύπτουν όλο το BitTorrent, αφού οι peers
# ακούν συχνά σε τυχαίες υψηλές θύρες, αλλά ένα μη μηδενικό ποσοστό εδώ
# είναι αρκετά ισχυρή ένδειξη ότι ο host συμμετέχει σε swarm.
#   6881-6889: BitTorrent (κλασικό εύρος)
#   6969:      BitTorrent tracker
#   51413:     Transmission (προεπιλογή)
P2P_PORTS = frozenset(range(6881, 6890)) | {6969, 51413}

# --- Χρονικά χαρακτηριστικά -----------------------------------------------
# Οι ώρες που μετρούν ως εργάσιμες για το Business_Ratio: 08:00-17:59.
BUSINESS_HOURS = (8, 18)

# Ημέρες που δεν καλύπτουν όλο το 24ωρο. Η 26/11/2025 ξεκινά στις 08:36,
# αλλά η ώρα 08 έχει μόλις 60 ροές από τα 2,3 εκατομμύρια της ημέρας, οπότε
# ουσιαστικά της λείπουν οι ώρες πριν τις 09:00.
#
# Δεν την εξαιρώ: 8.376 hosts (6,8% του συνόλου) εμφανίζονται
# μόνο εκείνη την ημέρα και θα έμεναν χωρίς κανένα χρονικό χαρακτηριστικό,
# δηλαδή θα σχημάτιζαν μια τεχνητή ομάδα. Το τίμημα είναι ότι λείπει η
# νυχτερινή δραστηριότητα της 26/11 και οι λόγοι ωρών για τους hosts εκείνης
# της ημέρας γέρνουν ελαφρώς προς το ωράριο.
# Βάλε [20251126] για να τρέξει το αντίθετο σενάριο.
PARTIAL_DAYS = []

# 5. Ρυθμίσεις επεξεργασίας
CHUNK_SIZE = 100000

# 6. Δυναμική Διαχείριση Μονοπατιών (Paths)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "raw_data": os.path.join(BASE_DIR, 'data', '1_raw'),
    "processed_data": os.path.join(BASE_DIR, 'data', '2_processed'),
    "bidirectional_data": os.path.join(BASE_DIR, 'data', '3_bidirectional'),
    "features": os.path.join(BASE_DIR, 'data', '4_profiles'),
    "results_macro": os.path.join(BASE_DIR, 'results', 'macro_clustering'),
    "results_micro": os.path.join(BASE_DIR, 'results', 'micro_clustering'),
}

# Δημιουργία των φακέλων αν δεν υπάρχουν (αλλιώς το to_csv αποτυγχάνει)
for _path in PATHS.values():
    os.makedirs(_path, exist_ok=True)
