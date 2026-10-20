import os
import pandas as pd
import math


def get_total_lines(file_path):
    """Μετράει αστραπιαία τις συνολικές γραμμές ενός αρχείου με ελάχιστη μνήμη."""
    with open(file_path, 'rb') as f:
        return sum(1 for _ in f)


def process_in_chunks(file_path, chunk_size, process_function, output_path=None, **kwargs):
    """
    Διαβάζει το αρχείο σε chunks και εφαρμόζει την process_function σε καθένα.

    - Αν δοθεί output_path, κάθε chunk γράφεται απευθείας στο αρχείο (η μνήμη μένει σταθερή)
      και επιστρέφεται το πλήθος των γραμμών που γράφτηκαν.
    - Αλλιώς, τα chunks συγκεντρώνονται και επιστρέφεται ένα DataFrame.
    """
    # Αφαιρούμε τις δικές μας μεταβλητές με ασφάλεια, για να μην πάνε στο Pandas!
    current_file = kwargs.pop('current_file', 1)
    total_files = kwargs.pop('total_files', 1)

    total_rows = get_total_lines(file_path)
    total_chunks = math.ceil(total_rows / chunk_size)

    print(f"      [Memory Manager] Γραμμές: {total_rows} | Chunks: {total_chunks}")

    if output_path and os.path.exists(output_path):
        os.remove(output_path)  # να μην μείνουν υπολείμματα από προηγούμενη εκτέλεση

    processed_chunks = []
    rows_written = 0
    first_write = True
    chunk_counter = 1

    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, **kwargs):
        file_progress = (chunk_counter / total_chunks) * 100
        overall_progress = ((current_file - 1) * 100 + file_progress) / total_files

        print(f"      -> Chunk {chunk_counter}/{total_chunks} "
              f"| Πρόοδος Αρχείου: {file_progress:.1f}% "
              f"| Συνολική Πρόοδος: {overall_progress:.1f}%")

        clean_chunk = process_function(chunk)
        if not clean_chunk.empty:
            if output_path:
                clean_chunk.to_csv(output_path, mode='w' if first_write else 'a',
                                   header=first_write, index=False)
                first_write = False
                rows_written += len(clean_chunk)
            else:
                processed_chunks.append(clean_chunk)

        chunk_counter += 1

    if output_path:
        return rows_written

    print("      [Memory Manager] Συνένωση όλων των επεξεργασμένων chunks...")
    return pd.concat(processed_chunks, ignore_index=True) if processed_chunks else pd.DataFrame()
