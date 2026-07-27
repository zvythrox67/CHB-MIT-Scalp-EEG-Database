# CHB-MIT Seizure Detection Pipeline

A machine learning pipeline that detects epileptic seizures from pediatric scalp EEG recordings. Loads raw EEG, filters it, splits it into windows, extracts time-domain, frequency-domain, entropy, and Hjorth features, then trains a Random Forest classifier to distinguish seizure from non-seizure activity.

Supports 24 patients, automatically switches to proper patient-independent evaluation (GroupKFold) once you have 2+ patients downloaded.

## Data Source

This project uses the **CHB-MIT Scalp EEG Database**, EEG recordings from 22 pediatric patients with intractable seizures, collected at Children's Hospital Boston and made publicly available through PhysioNet:

https://physionet.org/content/chbmit/1.0.0/

### Citation


> Guttag, J. (2010). CHB-MIT Scalp EEG Database (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C2K01R

> Shoeb, A. (2009). *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment*. PhD Thesis, Massachusetts Institute of Technology.

> Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. *Nature Health*. https://doi.org/10.1038/s44360-026-00096-z

## How to Setup:

1. Put this folder next to your `CHB-MIT Scalp EEG Database` folder, so the structure looks like:
   ```
   seizure_project/
       run_pipeline.py
       features.py
       summary_parser.py
       requirements.txt
   CHB-MIT Scalp EEG Database/
       chb01/
           chb01_01.edf
           chb01_02.edf
           ...
           chb01-summary.txt
       chb02/
           chb02_01.edf
           ...
           chb02-summary.txt
   ```
   Each patient gets its own folder with its own `.edf` files and its own `-summary.txt` — that's exactly how PhysioNet ships it, so downloading more patients into the same root folder just works with no code changes.

   (Or edit `ROOT_DIR` at the top of `run_pipeline.py` to point wherever your `CHB-MIT Scalp EEG Database` folder actually lives — it should point at the folder that CONTAINS chb01, chb02, etc., not at chb01 itself.)

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run it:
   ```
   python run_pipeline.py
   ```

## What it does

- Auto-detects every `chbNN` patient folder under `ROOT_DIR`
- Loads every `.edf` file for every patient found
- Filters out 60 Hz power line noise and keeps the 0.5–45 Hz band
- Splits each recording into 4-second windows
- Labels each window seizure/non-seizure using that patient's `-summary.txt`
- Extracts ~13 features per channel (time-domain, frequency band power, entropy, Hjorth) — with 23 channels that's roughly 300 features per window
- Saves the full feature table to `features_all.csv`, tagged with patient ID
- Trains and evaluates a Random Forest:
  - **2+ patients found:** uses `GroupKFold` grouped by patient — no patient's windows appear in both train and test within a fold. This is the patient-independent evaluation researchers expect.
  - **Only 1 patient found:** falls back to a chronological split by recording file (train on earlier files, test on later ones) — not as rigorous as GroupKFold, but avoids leaking windows from the same seizure event across train/test.
- Prints precision, recall, F1 (averaged across folds when using GroupKFold), confusion matrices, and the top 15 most important features

## First run will likely be slow-ish

Feature extraction (especially sample entropy) is the slowest step, and it scales with number of patients. If each patient has ~40 one-hour files, expect a few minutes per patient. If it's painfully slow, drop `sample_entropy` from `features.py` first — it's the most expensive one.

## Next steps once this runs

- Check `features_all.csv` to make sure labels look right (should be a small minority of rows = seizure) and that `patient_id` is populated correctly
- If F1 is very low or precision/recall are 0, sanity-check `summary_parser.py` against your actual `-summary.txt` files — seizure interval parsing is the most common bug
- Try 2-second windows vs 4-second windows
- Try SVM or XGBoost instead of Random Forest
- Compare single-patient (patient-specific) performance vs the pooled GroupKFold performance once you have several patients — that's a genuinely interesting research question in its own right
