import psutil
import pandas as pd
import numpy as np

#* --- 1. ΟΡΙΣΜΟΙ ΔΕΔΟΜΕΝΩΝ ---

# Τα ονόματα των στηλών για τα RAW αρχεία (χρήσιμο για το Script 01)
COLUMN_NAMES = [
    "Event",
    "XEvent",
    "Protocol",
    "Src_IP",
    "Src_Port",
    "Dst_IP",
    "Dst_Port",
    "XSrc_IP",
    "XSrc_Port",
    "XDst_IP",
    "XDst_Port",
    "Bytes_In",
    "Bytes_Out",
]

# Τύποι δεδομένων για τα Processed αρχεία (χρήσιμο για Scripts 02, 03)
DTYPES = {
    "Src_IP": "string",
    "Dst_IP": "string",
    "Protocol": "string",
    "Bytes_In": "object",  
    "Bytes_Out": "object",
}

#*--- 2. ΚΟΙΝΕΣ ΣΥΝΑΡΤΗΣΕΙΣ ---

# Υπολογίζει το βέλτιστο μέγεθος chunk βάσει της διαθέσιμης μνήμης.
def get_optimal_chunk_size(file_path, column_names=None, safety_factor=0.15):
    try:
        mem = psutil.virtual_memory()
        available_ram = mem.available
        
        # Ρυθμίσεις ανάγνωσης δείγματος
        kwargs = {'low_memory': False, 'nrows': 2000}
        
        # Αν μας δώσουν ονόματα στηλών, σημαίνει ότι είναι Raw αρχείο (χωρίς header)
        if column_names:
            kwargs['header'] = None
            kwargs['names'] = column_names
        
        sample = pd.read_csv(file_path, **kwargs)
        
        sample_memory_bytes = sample.memory_usage(deep=True).sum()
        bytes_per_row = sample_memory_bytes / 2000
        
        target_chunk_memory = available_ram * safety_factor
        optimal_size = int(target_chunk_memory / bytes_per_row)
        
        return max(10000, min(optimal_size, 2000000))
    except Exception:
        return 500000

# Μετατρέπει τα MikroTik strings (π.χ. '5.7 M', '240 B', '1.2 k') σε καθαρούς αριθμούς.
def parse_mikrotik_bytes(val):
    if pd.isna(val): return 0.0
    s = str(val).strip().upper().replace(',', '.')
    
    try:
        return float(s)
    except ValueError:
        pass
    
    multiplier = 1.0
    if 'M' in s: multiplier = 1_000_000.0; s = s.replace('M', '').replace('B', '')
    elif 'K' in s: multiplier = 1_000.0; s = s.replace('K', '').replace('B', '')
    elif 'G' in s: multiplier = 1_000_000_000.0; s = s.replace('G', '').replace('B', '')
    elif 'B' in s: s = s.replace('B', '')
        
    try:
        clean_num = "".join(filter(lambda x: x.isdigit() or x == '.', s))
        return float(clean_num) * multiplier
    except:
        return 0.0