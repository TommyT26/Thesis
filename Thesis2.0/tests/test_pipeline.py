"""
Έλεγχοι των σημείων όπου ένα λάθος θα περνούσε ΑΘΟΡΥΒΑ στα αποτελέσματα.

Δεν ελέγχω τα πάντα. Ελέγχω τις συναρτήσεις όπου ένα σφάλμα δεν ρίχνει το
pipeline αλλά αλλάζει σιωπηλά τους αριθμούς της αναφοράς — αυτά που δεν θα τα
έπιανα κοιτώντας την έξοδο. Τρία από αυτά έχουν ήδη συμβεί:

  - το Out Byte διαβαζόταν ως «Packets» και περνούσε στα μοντέλα ως σταθερά
    μηδενικό χαρακτηριστικό (νεκρή διάσταση στις αποστάσεις),
  - στην αντιστοίχιση ετικετών, όταν μια ομάδα της αναφοράς έλειπε από μια
    ημέρα, οι hosts εκείνης της ημέρας γίνονταν κενά και το ARI άλλαζε,
  - η ημερομηνία ως float32 στρογγυλοποιούνταν και ημέρες συγχωνεύονταν.

Κάθε ένα από αυτά έχει εδώ τον έλεγχο που θα το είχε πιάσει.

Εκτέλεση από τη ρίζα του project:

    py -m unittest discover -s tests -v
"""
import io
import os
import shutil
import tempfile
import contextlib
import unittest
from unittest import mock

# Το config απαιτεί κλειδί. Αν δεν υπάρχει .env (π.χ. σε καθαρό μηχάνημα),
# βάζω ένα δοκιμαστικό ΠΡΙΝ εισαχθεί οτιδήποτε από το src.
os.environ.setdefault('THESIS_SALT_KEY', 'test-key-only-for-unit-tests')

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, INTERNAL_SUBNET
from src.utils.anonymize import anonymize_ip, anonymize_series, get_internal_hashes
from src.utils.scaling import scale_features
from src.preprocessing.aggregate_flows import (normalize_protocol, add_time_columns,
                                               host_time_table)
from src.models.dbscan_clustering import deduplicate, resolve_min_samples
from src.models.daily_stability import align_day
from src.models.peer_recurrence import recurrence, consecutive_share
from src.models.hourly_profile import hourly_shares, partial_days
from src.models import port_semantics
from src.models.port_semantics import NO_PORTS, port_pair_cases


def profile_frame(rows):
    """Ελάχιστο DataFrame προφίλ με όλες τις στήλες που θέλει το scale_features."""
    frame = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = 1.0
    return frame


class TestAnonymize(unittest.TestCase):

    def test_ιδια_ip_δινει_ιδιο_hash(self):
        self.assertEqual(anonymize_ip('192.168.7.10'), anonymize_ip('192.168.7.10'))

    def test_διαφορετικες_ip_δινουν_διαφορετικα_hash(self):
        self.assertNotEqual(anonymize_ip('192.168.7.10'), anonymize_ip('192.168.7.11'))

    def test_το_hash_δεν_περιεχει_την_ip(self):
        # Αν η ανωνυμοποίηση σπάσει και επιστρέψει την είσοδο, θα φανεί εδώ.
        self.assertNotIn('192.168', anonymize_ip('192.168.7.10'))
        self.assertEqual(len(anonymize_ip('192.168.7.10')), 16)

    def test_series_συμφωνει_με_μεμονωμενο(self):
        ips = pd.Series(['10.0.0.1', '10.0.0.2', '10.0.0.1'])
        got = anonymize_series(ips)
        self.assertEqual(list(got), [anonymize_ip(ip) for ip in ips])

    def test_εσωτερικα_hashes_ειναι_254_μοναδικα(self):
        hashes = get_internal_hashes()
        self.assertEqual(len(hashes), 254)
        self.assertIn(anonymize_ip(f'{INTERNAL_SUBNET}1'), hashes)
        self.assertNotIn(anonymize_ip('8.8.8.8'), hashes)


class TestScaling(unittest.TestCase):

    def test_σχημα_και_κανονικοποιηση(self):
        frame = profile_frame({'Bytes_In': [1, 10, 100, 1000],
                               'Total_Flows': [1, 2, 3, 400]})
        X = scale_features(frame)
        self.assertEqual(X.shape, (4, len(FEATURE_COLUMNS)))
        varying = X[:, 0]
        self.assertAlmostEqual(float(varying.mean()), 0.0, places=9)
        self.assertAlmostEqual(float(varying.std()), 1.0, places=9)

    def test_σταθερο_χαρακτηριστικο_αναφερεται(self):
        # Ο έλεγχος που θα είχε πιάσει το Total_Packets (πάντα 0).
        frame = profile_frame({'Bytes_In': [1, 10, 100], 'Total_Flows': [0, 0, 0]})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            scale_features(frame)
        self.assertIn('Total_Flows', out.getvalue())
        self.assertIn('σταθερά', out.getvalue())

    def test_λειπον_χαρακτηριστικο_δινει_καθαρο_σφαλμα(self):
        frame = profile_frame({'Bytes_In': [1, 2]}).drop(columns=['Active_Hours'])
        with self.assertRaises(KeyError) as ctx:
            scale_features(frame)
        self.assertIn('Active_Hours', str(ctx.exception))


class TestAggregate(unittest.TestCase):

    def test_πρωτοκολλα_και_αριθμητικοι_κωδικοι(self):
        got = normalize_protocol(pd.Series(['tcp', 'TCP', '6', 'udp', '17', 'icmp', '1']))
        self.assertEqual(list(got),
                         ['TCP', 'TCP', 'TCP', 'UDP', 'UDP', 'OTHER', 'OTHER'])

    def test_ημερομηνια_δεν_χανει_ακριβεια(self):
        # Ο έλεγχος για το float32: το 20251203 δεν χωράει σε 24 bits mantissa.
        frame = pd.DataFrame({'Time': ['2025-12-03 23:59:36.210',
                                       '2025-12-02 00:00:01.000']})
        out, invalid = add_time_columns(frame)
        self.assertEqual(invalid, 0)
        self.assertEqual(list(out['Date']), [20251203, 20251202])
        self.assertEqual(list(out['Hour']), [23, 0])
        self.assertEqual(out['Date'].dtype.name, 'Int32')

    def test_ολα_τα_χρονικα_ακυρα_δινει_σφαλμα(self):
        with self.assertRaises(ValueError):
            add_time_columns(pd.DataFrame({'Time': ['οχι ημερομηνια', 'ουτε αυτο']}))

    def test_καθε_ροη_μετραει_και_για_τα_δυο_ακρα(self):
        frame = pd.DataFrame({'Src_Host': ['a'], 'Dst_Host': ['b'],
                              'Date': [20251127], 'Hour': [10],
                              'DayOfWeek': [3], 'Bytes': [500]})
        table = host_time_table(frame)
        self.assertEqual(set(table['Host']), {'a', 'b'})
        self.assertEqual(list(table['Flows']), [1, 1])
        self.assertEqual(list(table['Bytes']), [500, 500])


class TestDedup(unittest.TestCase):

    def test_ταυτοσημα_σημεια_παιρνουν_βαρος(self):
        X = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
        uniq, inverse, weights = deduplicate(X)
        self.assertEqual(len(uniq), 2)
        self.assertEqual(sorted(weights.tolist()), [1.0, 3.0])
        # Το inverse πρέπει να ανασυνθέτει το αρχικό σύνολο.
        np.testing.assert_allclose(uniq[inverse], X)

    def test_min_samples_κλιμακωνεται_και_φρεναρει(self):
        self.assertEqual(resolve_min_samples(64), 5)        # κάτω όριο
        self.assertEqual(resolve_min_samples(121_222), 182)  # αναλογικό
        self.assertEqual(resolve_min_samples(10_000_000), 500)  # άνω όριο


class TestAlignment(unittest.TestCase):
    """Η αντιστοίχιση ετικετών: εδώ ζούσε ένα πραγματικό σφάλμα."""

    def test_αντιστρεφει_μεταθεση(self):
        hosts = [f'h{i}' for i in range(6)]
        reference = pd.Series([0, 0, 1, 1, 2, 2], index=hosts)
        # Ίδια ομαδοποίηση, άλλη αρίθμηση: 0->2, 1->0, 2->1
        day = pd.Series([2, 2, 0, 0, 1, 1], index=hosts)
        translated, common, agreement, _ = align_day(day, reference)
        self.assertEqual(common, 6)
        self.assertEqual(agreement, 1.0)
        self.assertEqual(list(translated), list(reference))

    def test_δεν_παραγονται_ετικετες_εκτος_αναφορας(self):
        # ΤΟ ΣΦΑΛΜΑ, στην πραγματική του μορφή.
        #
        # Ο πίνακας επικάλυψης παίρνει στήλες μόνο από τις ετικέτες που
        # εμφανίζονται στους ΚΟΙΝΟΥΣ hosts. Εδώ η ομάδα 0 της αναφοράς ζει
        # μόνο στον h5, που λείπει από την ημέρα, οπότε χωρίς το reindex ο
        # πίνακας βγαίνει 3x2: τρεις ομάδες ημέρας, δύο στήλες. Η τρίτη μένει
        # αταίριαστη και παίρνει «εφεδρικό» αριθμό max+1 = 3 — ετικέτα που
        # ΔΕΝ υπάρχει στη συνολική ομαδοποίηση. Το αποτέλεσμα είναι ομάδα
        # φάντασμα στα γραφήματα σύνθεσης και στους πίνακες συμφωνίας, ενώ
        # η σωστή απάντηση ήταν να αντιστοιχιστεί στην ομάδα 0.
        reference = pd.Series({'h0': 1, 'h1': 1, 'h2': 2,
                               'h3': 2, 'h4': 2, 'h5': 0})
        day = pd.Series({'h0': 0, 'h1': 0, 'h2': 1, 'h3': 1, 'h4': 2})

        translated, _, _, _ = align_day(day, reference)
        self.assertEqual(translated.isna().sum(), 0)
        self.assertTrue(
            set(translated.dropna()).issubset(set(reference.unique())),
            f"ετικέτες εκτός αναφοράς: {set(translated.dropna())} "
            f"έναντι {set(reference.unique())}")

    def test_επιπλεον_ομαδα_ημερας_παιρνει_νεο_αριθμο(self):
        hosts = [f'h{i}' for i in range(4)]
        reference = pd.Series([0, 0, 1, 1], index=hosts)
        day = pd.Series([0, 1, 2, 2], index=hosts)
        translated, _, _, _ = align_day(day, reference)
        self.assertEqual(translated.isna().sum(), 0)
        # Καμία ετικέτα δεν χάνεται με το να στριμωχτεί πάνω σε άλλη.
        self.assertEqual(len(set(translated)), 3)


class TestRecurrence(unittest.TestCase):

    def test_λογος_επαναληψης(self):
        frame = pd.DataFrame({
            'Host': ['a', 'a', 'a', 'b', 'b'],
            'Service_Port': [80, 80, 443, 22, 23],
            'Ημέρα': [1, 2, 1, 1, 2],
        })
        table = recurrence(frame, 'Service_Port')
        # Ο 'a' είδε δύο θύρες, η μία σε δύο ημέρες -> 1/2
        self.assertEqual(table.loc['a', 'Πλήθος'], 2)
        self.assertAlmostEqual(table.loc['a', 'Λόγος_Επανάληψης'], 0.5)
        # Ο 'b' είδε δύο θύρες, καμία δύο φορές -> 0
        self.assertAlmostEqual(table.loc['b', 'Λόγος_Επανάληψης'], 0.0)

    def test_ποσοστο_διαδοχικων_θυρων(self):
        frame = pd.DataFrame({
            'Host': ['scan'] * 4 + ['service'] * 3,
            'Service_Port': [80, 81, 82, 83, 22, 443, 8080],
            'Ημέρα': [1] * 7,
        })
        shares = consecutive_share(frame, {'scan', 'service'})
        self.assertAlmostEqual(shares['scan'], 1.0)     # 80,81,82,83
        self.assertAlmostEqual(shares['service'], 0.0)  # διάσπαρτες

    def test_ενα_μονο_στοιχειο_δεν_οριζεται(self):
        frame = pd.DataFrame({'Host': ['x'], 'Service_Port': [80], 'Ημέρα': [1]})
        shares = consecutive_share(frame, {'x'})
        self.assertTrue(np.isnan(shares['x']))


class TestPortSemantics(unittest.TestCase):
    """Το Service_Port == 0: πρωτόκολλο χωρίς θύρες, όχι θύρα υπηρεσίας."""

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        pd.DataFrame({
            # δύο ροές ICMP/IGMP (θύρα 0 και στις δύο πλευρές), που ΔΕΝ πρέπει
            # να μετρηθούν ως «δύο καταχωρημένες θύρες»
            'Src_Port': [0, 0, 443, 50000, 22],
            'Dst_Port': [0, 0, 51000, 52000, 80],
        }).to_csv(os.path.join(self.folder, 'clean_20251127.csv'), index=False)

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_οι_ροες_χωρις_θυρες_μετριουνται_χωριστα(self):
        # ΤΟ ΣΦΑΛΜΑ: με fillna(-1) και σκέτο «< 1024», η θύρα 0 του ICMP/IGMP
        # περνούσε ως καταχωρημένη και οι δύο αυτές ροές έμπαιναν στο «δύο
        # γνωστές θύρες», φουσκώνοντας ακριβώς τον αριθμό που μπαίνει στην
        # αναφορά. Εδώ πρέπει να βγουν σε δική τους γραμμή.
        with mock.patch.dict(port_semantics.PATHS, {'processed_data': self.folder}):
            table, total, with_ports = port_pair_cases()

        self.assertEqual(total, 5)
        self.assertEqual(with_ports, 3)
        counts = dict(zip(table['Περίπτωση'], table['Ροές']))
        self.assertEqual(counts[NO_PORTS], 2)
        self.assertEqual(counts['Δύο γνωστές θύρες'], 1)   # 22 / 80
        self.assertEqual(counts['Μία γνωστή θύρα'], 1)     # 443 / 51000
        self.assertEqual(counts['Καμία γνωστή θύρα'], 1)   # 50000 / 52000

    def test_τα_ποσοστα_ειναι_επι_των_ροων_με_θυρες(self):
        # Αλλιώς τα τρία ποσοστά δεν αθροίζουν σε 1 και ο πίνακας της
        # αναφοράς διαβάζεται λάθος.
        with mock.patch.dict(port_semantics.PATHS, {'processed_data': self.folder}):
            table, total, with_ports = port_pair_cases()
        rest = table[table['Περίπτωση'] != NO_PORTS]
        # places=3 και όχι 6: τα ποσοστά στρογγυλοποιούνται σε 4 δεκαδικά για
        # το CSV, οπότε τρία τρίτα δίνουν 0,9999 και όχι ακριβώς 1.
        self.assertAlmostEqual(float(rest['Ποσοστό'].sum()), 1.0, places=3)
        portless = table.loc[table['Περίπτωση'] == NO_PORTS, 'Ποσοστό'].iloc[0]
        self.assertAlmostEqual(float(portless), 2 / 5, places=6)


class TestHourly(unittest.TestCase):

    def test_τεραστιος_host_δεν_κυριαρχει_στην_ομαδα(self):
        # Η κανονικοποίηση ανά host: αλλιώς η καμπύλη της ομάδας θα ήταν η
        # καμπύλη της πύλης, που έχει 18,3 εκατ. ροές.
        rows = []
        for hour in range(24):
            rows.append(('flat', hour, 10))
            rows.append(('GIANT', hour, 100_000))
        rows.append(('spiky', 9, 1_000))  # ένας host με όλη την κίνηση στις 09:00
        frame = pd.DataFrame(rows, columns=['Host', 'Hour', 'Flows'])
        labels = pd.Series({'flat': 0, 'GIANT': 0, 'spiky': 0})

        curve, hosts = hourly_shares(frame, labels)
        self.assertEqual(len(hosts), 3)
        # Δύο επίπεδοι (1/24 ο καθένας) και ένας με όλα στις 09:00:
        # μέσος όρος στις 09:00 = (1/24 + 1/24 + 1)/3
        self.assertAlmostEqual(curve.loc[0, 9], (2 / 24 + 1) / 3, places=9)
        # Αν κυριαρχούσε ο GIANT, η καμπύλη θα ήταν επίπεδη στο 1/24.
        self.assertGreater(curve.loc[0, 9], 0.3)
        # Κάθε καμπύλη ομάδας αθροίζει σε 1.
        self.assertAlmostEqual(float(curve.loc[0].sum()), 1.0, places=9)

    def test_μερικη_ημερα_εντοπιζεται_απο_τα_δεδομενα(self):
        frame = pd.DataFrame({
            'Host': ['a'] * 4,
            'Date': [20251126, 20251126, 20251127, 20251127],
            'Hour': [9, 23, 0, 23],
            'DayOfWeek': [2, 2, 3, 3],
            'Flows': [1, 1, 1, 1],
        })
        found, _ = partial_days(frame)
        self.assertIn(20251126, found)
        self.assertNotIn(20251127, found)


if __name__ == '__main__':
    unittest.main()
