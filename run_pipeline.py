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

