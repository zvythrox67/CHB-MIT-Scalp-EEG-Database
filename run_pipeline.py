import re
from pathlib import Path 
import numpy as np
import pandas as pd
import mne
from summary_parser import parse_summary_file
from features import extract_window_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

mne.set_log_level("ERROR")

ROOT_DIR = Path("./CHB-MIT Scalp EEG Database")

WINDOW_SECONDS = 4
WINDOW_OVERLAP = 0.0 

OUTPUT_CSV = "features_all.csv"
N_TEST_FILES_SINGLE_PATIENT = 3
N_SPLITS = 5

def find_patient_folders(root_dir):
    pattern = re.compile(r"^chb\d+$")
    patient_dirs = [p for p in root_dir.iterdir() if p.is_dir() and pattern.match(p.name)]
    patient_dirs.sort(key=lambda p: p.name)
    
    if not patient_dirs:
        raise FileNotFoundError(
            f"No chbNN folders found under {root_dir}. "
            f"Check ROOT_DIR points at the folder that CONTAINS chb01, chb02, etc."
        )
    return {p.name: p for p in patient_dirs}

def preprocess_raw(raw):
    raw.notch_filter(freqs = 60, verbose = False)
    raw.filter(l_freq=0.5, h_freq=45, verbose=False)
    return raw

def window_and_label(raw, seizure_intervals, window_seconds, overlap):
    sfreq = raw.info["sfreq"]
    data = raw.get_data()
    n_channels, n_samples = data.shape
    
    win_samples = int(window_seconds * sfreq)
    step_samples = int(win_samples * (1 - overlap)) if overlap > 0 else win_samples
    step_samples = max(step_samples, 1)
    
    windows = []
    labels = []
    
    for start_sample in range(0, n_samples - win_samples + 1, step_samples):
        end_sample = start_sample + win_samples
        start_sec = start_sample / sfreq
        end_sec = end_sample / sfreq
        
        is_seizure = 0
        for (sz_start, sz_end) in seizure_intervals:
            if start_sec < sz_end and end_sec > sz_start:
                is_seizure = 1
                break
            
        windows.append(data[:, start_sample:end_sample])
        labels.append(is_seizure)
    
    return windows, labels

def process_patient(patient_id, patient_dir):
    summary_file = patient_dir / f"{patient_id}-summary.txt"
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_file}")
    
    seizures_by_file = parse_summary_file(summary_file)
    edf_files = sorted(patient_dir.glob(f"{patient_id}_*.edf"))
    
    if not edf_files:
        raise FileNotFoundError(f"No .edf files found in {patient_dir}")
    
    rows = []
    
    for edf_path in edf_files:
        fname = edf_path.name
        print(f" Processing {fname}...")
        
        raw = mne.io.read_raw_edf(edf_path, preload = True, verbose = False)
        preprocess_raw(raw)
        
        seizure_intervals = seizures_by_file.get(fname, [])
        windows, labels = window_and_label(raw, seizure_intervals, WINDOW_SECONDS, WINDOW_OVERLAP)
        
        sfreq = raw.info["sfreq"]
        channel_names = raw.info["ch_names"]
        
        for window, label in zip(windows, labels):
            feats = extract_window_features(window, sfreq, channel_names)
            feats["label"] = label
            feats["source_file"] = fname
            feats["patient_id"] = patient_id
            rows.append(feats)
        
        n_seizure_windows = sum(labels)
        print(f"    -> {len(windows)} windows, {n_seizure_windows} labeled seizure")
        
    return rows

def build_feature_table():
    patient_folders = find_patient_folders(ROOT_DIR)
    print(f"Found {len(patient_folders)} patient(s): {list(patient_folders.keys())}\n")

    all_rows = []
    for patient_id, patient_dir in patient_folders.items():
        print(f"Patient {patient_id}:")
        all_rows.extend(process_patient(patient_id, patient_dir))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved feature table: {OUTPUT_CSV} ({df.shape[0]} rows, {df.shape[1]} columns)")
    return df

def chronological_train_test_split(df, n_test_files):
    file_names_in_order = list(dict.fromkeys(df["source_file"]))
    n_test_files = min(n_test_files, max(len(file_names_in_order) - 1, 1))
    test_files = set(file_names_in_order[-n_test_files:])
    
    train_df = df[~df["source_file"].isin(test_files)]
    test_df = df[df["source_file"].isin(test_files)]

    print(f"\nTrain files: {sorted(set(train_df['source_file']))}")
    print(f"Test files:  {sorted(test_files)}")

    return [(train_df, test_df)]

