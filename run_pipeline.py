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