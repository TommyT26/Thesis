import hmac
import hashlib
import pandas as pd

from src.config import SALT_KEY_BYTES, INTERNAL_SUBNET, SUBNET_PREFIXES


def anonymize_ip(ip_str):
    """HMAC-SHA256 της IP, κρατάμε τους πρώτους 16 hex χαρακτήρες."""
    if pd.isna(ip_str) or not isinstance(ip_str, str):
        return ip_str
    return hmac.new(SALT_KEY_BYTES, ip_str.encode('utf-8'), hashlib.sha256).hexdigest()[:16]


def anonymize_series(series):
    """Ανωνυμοποιεί μια στήλη υπολογίζοντας το hash μία φορά ανά μοναδική IP (πολύ πιο γρήγορο από apply)."""
    series = series.astype(object)
    mapping = {ip: anonymize_ip(ip) for ip in series.dropna().unique()}
    return series.map(mapping)


# Τα οκτέτα που είναι HOSTS. Το .0 είναι η διεύθυνση του δικτύου και το .255
# η διεύθυνση εκπομπής: καμία από τις δύο δεν είναι μηχανή. Ίδιο εύρος με το
# get_internal_hashes, και αυτό δεν είναι σύμπτωση — αν τα υποδίκτυα
# κρατούσαν άλλο εύρος, το εργαστήριο θα ομαδοποιούσε 64 διευθύνσεις ενώ τα
# υπόλοιπα 65, δηλαδή το ένα θα είχε μέσα του τη διεύθυνση εκπομπής και το
# άλλο όχι. Στο 192.168.8.x αυτό ακριβώς συνέβαινε: η .255 έμπαινε ως host με
# Out_Ratio 0, TCP_Ratio 0 και 36 bytes ανά ροή, και σήκωνε 562.975 ροές,
# δηλαδή το 10% του υποδικτύου.
HOST_OCTETS = range(1, 255)


def get_subnet_hashes(prefix, last_octets=HOST_OCTETS):
    """
    Τα hashes των HOSTS ενός /24 (οκτέτα 1-254).

    Δεν αντιστρέφει hashes: τα ξαναϋπολογίζει ΠΡΟΣ ΤΑ ΕΜΠΡΟΣ από το πρόθεμα,
    με την ίδια συνάρτηση που ανωνυμοποίησε τα δεδομένα. Αυτό δουλεύει μόνο
    για υποδίκτυα που ξέρουμε ήδη, και είναι ακριβώς η ιδιότητα που κάνει το
    HMAC ψευδωνυμοποίηση και όχι ανωνυμοποίηση: όποιος κρατά το κλειδί
    μπορεί να ελέγξει μια υπόθεση για μια διεύθυνση. Το 192.168.0.0/16 έχει
    65.536 διευθύνσεις, οπότε η ιδιότητα δεν περιορίζεται στα δηλωμένα
    υποδίκτυα — το κλειδί είναι το μυστικό, όχι το εύρος.
    """
    return {anonymize_ip(f"{prefix}{octet}") for octet in last_octets}


def subnet_labels(prefixes, last_octets=range(256)):
    # ΠΡΟΣΟΧΗ στη διαφορά από το get_subnet_hashes: εδώ το εύρος είναι 0-255
    # επίτηδες. Η συνάρτηση δεν απαντά «ποιοι είναι οι hosts» αλλά «σε ποιο
    # υποδίκτυο ανήκει αυτή η διεύθυνση», και μια εκπομπή προς το
    # 192.168.8.255 είναι κίνηση προς το 192.168.8.x με κάθε έννοια. Το
    # χρησιμοποιούν οι πίνακες επαφών, θυρών και κατεύθυνσης· η ομαδοποίηση
    # χρησιμοποιεί το get_subnet_hashes.
    """
    hash -> ετικέτα υποδικτύου ('192.168.8.x'), για όλα τα δοσμένα προθέματα.

    Γυρίζει την ετικέτα του /24 και ΠΟΤΕ το τελευταίο οκτέτο, ώστε τα αρχεία
    των αποτελεσμάτων να μένουν χωρίς διευθύνσεις: η ομάδα «192.168.8.x»
    λέει ό,τι χρειάζεται η ανάλυση, ενώ το «192.168.8.195» θα έσπαγε την
    ψευδωνυμοποίηση σε κάθε CSV που μοιράζεται.
    """
    mapping = {}
    for prefix in prefixes:
        label = f"{prefix}x"
        for octet in last_octets:
            mapping[anonymize_ip(f"{prefix}{octet}")] = label
    return mapping


def label_subnets(series, prefixes, last_octets=range(256)):
    """Η στήλη των hashes ως ετικέτες υποδικτύου· NaN για ό,τι είναι εκτός."""
    return series.map(subnet_labels(prefixes, last_octets))


def _our_prefixes():
    """Τα υποδίκτυα που αποτελούν το δικό μας δίκτυο."""
    return list(SUBNET_PREFIXES) or [INTERNAL_SUBNET]


def get_internal_hashes():
    """
    Τα hashes των HOSTS του εσωτερικού δικτύου: όλα τα υποδίκτυα του
    SUBNET_PREFIXES, οκτέτα 1-254.

    Ήταν μόνο το INTERNAL_SUBNET. Οι μετρήσεις έδειξαν ότι 54 hosts από δύο
    ακόμη ιδιωτικά υποδίκτυα περνούσαν ως hosts του διαδικτύου: το
    192.168.8.x στέλνει το 83,9% της κίνησής του στο εργαστήριο και το
    192.168.1.x το 97,6%, ενώ προς τα έξω πάει 2,8% και 1,9%. Μια διεύθυνση
    RFC1918 δεν είναι host του διαδικτύου με καμία ανάγνωση.

    Οι διευθύνσεις δικτύου και εκπομπής ΔΕΝ είναι εδώ: δεν είναι μηχανές και
    δεν ομαδοποιούνται. Φεύγουν όμως και από τον εξωτερικό πληθυσμό, μέσω του
    get_network_hashes — δες το data_split.
    """
    hashes = set()
    for prefix in _our_prefixes():
        hashes |= get_subnet_hashes(prefix, HOST_OCTETS)
    return hashes


def get_network_hashes():
    """
    ΚΑΘΕ διεύθυνση των δικών μας υποδικτύων, 0-255: hosts, διευθύνσεις
    δικτύου και διευθύνσεις εκπομπής.

    Χρησιμεύει για να οριστεί τι ΔΕΝ είναι διαδίκτυο. Οι διευθύνσεις εκπομπής
    δεν είναι hosts, άρα δεν ομαδοποιούνται· αλλά δεν είναι ούτε εξωτερικές,
    και όσο έμεναν εκεί η 192.168.7.255 μετριόταν ως ανωμαλία του Isolation
    Forest.
    """
    hashes = set()
    for prefix in _our_prefixes():
        hashes |= get_subnet_hashes(prefix, range(256))
    return hashes
