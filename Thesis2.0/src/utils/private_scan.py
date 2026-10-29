"""
Ποιοι hosts των προφίλ είναι ιδιωτικές διευθύνσεις RFC1918.

ΓΙΑΤΙ ΧΡΕΙΑΖΕΤΑΙ ΞΕΧΩΡΙΣΤΟ ΒΗΜΑ. Μετά την ανωνυμοποίηση δεν υπάρχει IP, μόνο
HMAC-SHA256. Τα hashes δεν αντιστρέφονται, οπότε ο μόνος τρόπος να απαντηθεί
«είναι αυτός ο host ιδιωτική διεύθυνση;» είναι να υπολογιστεί προς τα εμπρός
το hash κάθε διεύθυνσης του RFC1918 και να ελεγχθεί αν εμφανίζεται στα
δεδομένα. Είναι 17.891.328 διευθύνσεις και περίπου 68 δευτερόλεπτα, οπότε δεν
μπορεί να γίνεται μέσα σε κάθε στάδιο που χωρίζει πληθυσμούς: γίνεται ΜΙΑ
φορά, γράφεται ο κατάλογος των ευρημάτων, και τα στάδια διαβάζουν αυτόν.

ΤΙ ΒΡΗΚΕ. Το πρώτο τρέξιμο έδειξε 85 ιδιωτικές διευθύνσεις εκτός των τριών
υποδικτύων της ανάλυσης, με 186.178 ροές (0,73% του συνόλου), από τις οποίες
20 ήταν μέσα στις 1.846 ανωμαλίες του Isolation Forest. Δεν είναι θόρυβος:
κίνηση OpenFlow (6653/6633) προς το 192.168.8.x, DNS και NTP, Kubernetes API
(6443), γέφυρες Docker (172.17/172.18), Splashtop (6783) και MikroTik
Neighbor Discovery (5678). Είναι υποδομή, όχι hosts του διαδικτύου.

ΓΙΑΤΙ ΔΕΝ ΤΙΣ ΕΙΧΑΜΕ ΔΕΙ. Η πρώτη σάρωση κάλυπτε 192.168.0.0/16 και
172.16.0.0/12 και σταματούσε εκεί. Το 10.0.0.0/8 είναι το μεγαλύτερο μπλοκ
του RFC1918 και περιείχε μόνο του 56 διευθύνσεις με 116.851 ροές, δηλαδή
περισσότερες από τις άλλες δύο ζώνες μαζί.

ΑΣΦΑΛΕΙΑ. Ο κατάλογος κρατά hash και ετικέτα /24, ποτέ το τελευταίο οκτέτο —
ίδιος κανόνας με όλα τα άλλα αποτελέσματα. Η σάρωση γίνεται με streaming: δεν
κρατιέται ποτέ στη μνήμη σύνολο 17,9 εκατομμυρίων hashes, μόνο τα ~123.000
των προφίλ και τα λίγα ευρήματα. Καμία αντιστροφή, κανένας πίνακας
αντιστοίχισης δεν γράφεται στον δίσκο.
"""
import os
import csv

from src.config import PATHS, BASE_DIR, SUBNET_PREFIXES
from src.utils.anonymize import anonymize_ip

OUTPUT = os.path.join(BASE_DIR, 'results', 'private_hosts.csv')

# Τα τρία μπλοκ του RFC1918, ως πλήθος /24 το καθένα:
#   10.0.0.0/8      -> 65.536 υποδίκτυα /24
#   172.16.0.0/12   ->  4.096
#   192.168.0.0/16  ->    256
TOTAL_PREFIXES = 65536 + 4096 + 256


def private_prefixes():
    """Κάθε /24 του RFC1918, ως πρόθεμα με τελεία στο τέλος."""
    for a in range(256):
        for b in range(256):
            yield f'10.{a}.{b}.'
    for a in range(16, 32):
        for b in range(256):
            yield f'172.{a}.{b}.'
    for b in range(256):
        yield f'192.168.{b}.'


def scan(hosts, report=None, prefixes=None):
    """
    {hash: ετικέτα /24} για κάθε host των δεδομένων που είναι RFC1918.

    hosts: σύνολο των hashes που εμφανίζονται στα προφίλ.
    report: προαιρετική συνάρτηση προόδου, καλείται ανά 4.096 προθέματα.
    prefixes: ποια /24 να σαρωθούν· προεπιλογή όλο το RFC1918. Υπάρχει ως
    παράμετρος για τους ελέγχους, που δεν έχουν λόγο να πληρώνουν 17,9
    εκατομμύρια hashes για να επαληθεύσουν τη λογική της αντιστοίχισης.
    """
    found = {}
    for index, prefix in enumerate(private_prefixes() if prefixes is None else prefixes):
        for octet in range(256):
            host = anonymize_ip(prefix + str(octet))
            if host in hosts:
                found[host] = prefix + 'x'
        if report is not None and index % 4096 == 0:
            report(index, len(found))
    return found


def profile_hosts():
    """Τα hashes των προφίλ. Μόνο η στήλη Host, χωρίς pandas."""
    path = os.path.join(PATHS['features'], 'features_global.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Δεν βρέθηκε το {path}. Τρέξε πρώτα:  python main.py --step features")
    with open(path, encoding='utf-8-sig', newline='') as handle:
        return {row['Host'] for row in csv.DictReader(handle)}


def load():
    """
    Ο κατάλογος από τον δίσκο: {hash: ετικέτα /24}. Άδειος αν λείπει.

    Το άδειο δεν είναι σιωπηλό. Όποιος χωρίζει πληθυσμούς χωρίς αυτόν τον
    κατάλογο ξαναβάζει τις ιδιωτικές διευθύνσεις στους «εξωτερικούς hosts του
    διαδικτύου», που είναι ακριβώς το λάθος που το βήμα αυτό διορθώνει.
    """
    if not os.path.exists(OUTPUT):
        return {}
    with open(OUTPUT, encoding='utf-8-sig', newline='') as handle:
        return {row['Host']: row['Υποδίκτυο'] for row in csv.DictReader(handle)}


def ours():
    """Τα προθέματα των υποδικτύων της ανάλυσης, για να εξαιρεθούν."""
    return tuple(SUBNET_PREFIXES)


def others():
    """
    Τα hashes των ιδιωτικών διευθύνσεων ΕΚΤΟΣ των δικών μας υποδικτύων.

    Αυτό είναι το σύνολο που βγαίνει από τον εξωτερικό πληθυσμό. Τα δικά μας
    υποδίκτυα τα χειρίζεται το populations.py και δεν περνούν από εδώ.
    """
    mine = ours()
    return frozenset(host for host, label in load().items()
                     if not label.rstrip('x').startswith(mine))


def run():
    print("\n--- Έναρξη: Εντοπισμός Ιδιωτικών Διευθύνσεων (RFC1918) ---")
    hosts = profile_hosts()
    print(f" -> {len(hosts):,} hosts στα προφίλ")
    print(f" -> Σάρωση {TOTAL_PREFIXES * 256:,} διευθύνσεων του RFC1918 "
          f"(10/8, 172.16/12, 192.168/16)")
    print("    Υπολογίζονται hashes προς τα εμπρός· καμία αντιστροφή.")

    def report(index, hits):
        print(f"    - {index / TOTAL_PREFIXES:5.1%} | ευρήματα ως τώρα: {hits}", flush=True)

    found = scan(hosts, report=report)

    mine = ours()
    inside = {h: l for h, l in found.items() if l.rstrip('x').startswith(mine)}
    outside = {h: l for h, l in found.items() if h not in inside}

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Host', 'Υποδίκτυο', 'Στα_υποδίκτυα_της_αναλυσης'])
        for host, label in sorted(found.items(), key=lambda item: item[1]):
            writer.writerow([host, label, host in inside])

    print(f"\n -> Βρέθηκαν {len(found)} ιδιωτικές διευθύνσεις στα δεδομένα:")
    print(f"    {len(inside)} στα υποδίκτυα της ανάλυσης ({', '.join(mine)})")
    print(f"    {len(outside)} ΕΚΤΟΣ αυτών — βγαίνουν από τον εξωτερικό πληθυσμό")

    groups = {}
    for label in outside.values():
        groups[label] = groups.get(label, 0) + 1
    for label, count in sorted(groups.items(), key=lambda item: -item[1]):
        print(f"      {label:<18} {count}")

    print(f"\n -> Ο κατάλογος αποθηκεύτηκε στο: {OUTPUT}")
    print("--- Ολοκληρώθηκε ---")


if __name__ == "__main__":
    run()
