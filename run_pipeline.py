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