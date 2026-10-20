import hmac
import hashlib
import pandas as pd

from src.config import SALT_KEY_BYTES, INTERNAL_SUBNET


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


def get_internal_hashes():
    """Τα hashes των εσωτερικών IP του εργαστηρίου (ίδια συνάρτηση hash με την ανωνυμοποίηση)."""
    return {anonymize_ip(f"{INTERNAL_SUBNET}{i}") for i in range(1, 255)}
