# Solar Flare Forecasting

A comprehensive machine learning and statistical analysis framework for detecting and classifying solar flares using X-ray light curve data from the SoLEXS instrument and GOES satellite observations.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Dependencies](#dependencies)
5. [Data Processing Pipeline](#data-processing-pipeline)
6. [Detection Methods](#detection-methods)
7. [Classification System](#classification-system)
8. [Results](#results)
9. [Getting Started](#getting-started)
10. [Usage Guide](#usage-guide)
11. [Model Performance](#model-performance)

---

## Overview

This project implements multiple machine learning and statistical methods to detect solar flares in X-ray light curve (LC) data and classify them according to the official GOES (Geostationary Operational Environmental Satellite) classification system. The framework combines:

- **Statistical detectors** (Dual-window, CUSUM, and Wavelet-based methods)
- **Probabilistic models** (Bayesian likelihood ratio with Welch's t-test)
- **Machine learning classifiers** (Neural Networks, Random Forest, Gradient Boosting)

The project uses SoLEXS light curve data from the ISRO ISSDC Pradan website and cross-references detections with GOES-18 XRS flux measurements for official classification.

---

## Key Features

✅ **Multi-method flare detection** — Three independent statistical approaches with complementary strengths  
✅ **Real-time compatible** — Causal algorithms suitable for deployment  
✅ **Ground truth integration** — HEK (Heliophysics Event Knowledgebase) flare matching and validation  
✅ **GOES classification** — Official X-ray flux-based flare categorization (A, B, C, M, X classes)  
✅ **Per-class performance analysis** — Separate metrics for each flare magnitude  
✅ **Detection latency measurement** — Quantifies how quickly flares are flagged after onset  
✅ **Interactive visualization** — Light curve plots with detected flares and classifications  

---

## Project Structure

```
SolarFlareForecasting/
├── data/                          # Data directory
│   ├── hek_flares.csv             # Ground truth flare annotations (HEK)
│   ├── detections_train.csv       # Training detections from dual-window method
│   ├── cusum_detections_train.csv # Training detections from CUSUM method
│   └── wavelet_detections_train.csv # Training detections from Wavelet method
│
├── lc_files/                      # SoLEXS light curve FITS files (input)
├── pi_files/                      # SoLEXS PI (Pulse Invariant) FITS files (optional)
│
├── Data Processing Scripts
│   ├── unzip_solex.py             # Extract .lc and .pi files from ISRO downloads
│   ├── clean_lcfiles.py           # Remove problematic light curve files by date
│   ├── vis_lc.py                  # Visualize individual .lc FITS files
│   ├── vis_pi.py                  # Visualize individual .pi FITS files
│   └── solexs_flare_analysis.py   # Statistical analysis of light curves
│
├── Data Extraction & Preparation
│   ├── build_flare_dataset.py     # Extract features and build training dataset
│   ├── getFlares_data.py          # Fetch HEK flare annotations for matching dates
│   └── goes_classify.py           # Classify detected flares using GOES XRS data
│
├── Detection Models
│   ├── nn_flareclassifier.py      # Neural Network flare detector (30 epochs)
│   ├── rfc.ipynb                  # Random Forest Classifier (interactive notebook)
│   └── train_flare_models.ipynb   # RandomForest + HistGradientBoosting training
│
├── Visualization
│   ├── lc_flared.py               # Plot light curves with detected flares overlay
│   └── Results visualizations      # Output plots and analysis charts
│
└── README.md                       # This file
```

---

## Dependencies

Install the required packages:

```bash
pip install numpy pandas scipy matplotlib scikit-learn
pip install astropy sunpy sunpy[net]
pip install tensorflow  # For neural network models
```

**Core Libraries:**
- **NumPy / Pandas** — Data manipulation and analysis
- **SciPy** — Signal processing (find_peaks, Welch's method, CUSUM, wavelets)
- **Scikit-learn** — Machine learning classifiers (Random Forest, Gradient Boosting)
- **TensorFlow/Keras** — Neural network implementation
- **Astropy** — FITS file handling
- **SunPy** — Solar data access (GOES XRS data download via Fido)
- **Matplotlib** — Visualization

---

## Data Processing Pipeline

### 1. **Data Acquisition & Extraction** (`unzip_solex.py`)

```bash
python unzip_solex.py <source_dir> <target_dir>
```

- Downloads ISRO ISSDC Pradan SoLEXS light curve files (zipped)
- Extracts `.lc` (light curve) and `.pi` (pulse invariant) FITS files
- **Filters:** Only saves files with non-empty GTI (Good Time Interval) extensions
- Populates `lc_files/` directory for further processing

### 2. **Data Cleaning** (`clean_lcfiles.py`)

Removes problematic light curve files identified through manual review:
```bash
python clean_lcfiles.py --dates [YYYY-MM-DD, ...]
```

- Deletes corrupted or unreliable data files
- Prepares clean dataset for analysis

### 3. **Visualization & Inspection** (`vis_lc.py`, `vis_pi.py`)

```bash
python vis_lc.py <lc_file>
python vis_pi.py <pi_file>
```

- Plots raw X-ray light curves from SoLEXS
- Visualizes pulse invariant spectra
- Helps identify data quality issues and flare signatures

### 4. **Ground Truth Acquisition** (`getFlares_data.py`)

```bash
python getFlares_data.py
```

- Fetches HEK (Heliophysics Event Knowledgebase) flare annotations
- Selects dates matching available light curve files
- Outputs `hek_flares.csv` with:
  - Event ID, start time, peak time, end time
  - GOES class designation (A, B, C, M, X)
  - Flare location coordinates

### 5. **Feature Engineering** (`build_flare_dataset.py`)

```bash
python build_flare_dataset.py
```

Extracts features from light curve files and HEK annotations:

**Extracted Features:**
- **Temporal:** onset time, peak time, duration, rise time, decay time
- **Amplitude:** peak counts, maximum count rate, mean background
- **Statistical:** count variance, skewness, kurtosis, dynamic range
- **Signal characteristics:** flare energy (integral above baseline), fluence
- **Morphology:** rise slope, decay slope, asymmetry
- **Context:** time of day, solar rotation number, active region proximity

Output: Training/validation datasets with binary labels (flare/no-flare) and GOES class labels

---

## Detection Methods

### Method 1: **Threshold-Based Detection**

A simple baseline method that flags continuous intervals where:
- **Condition:** `counts > background + 5σ` AND duration `≥ 60 s`
- **Rationale:** 5σ cut gives ~1-in-3.5 million false positive rate per sample
- **Strength:** Fast, interpretable
- **Weakness:** Misses overlapping flares

### Method 2: **SciPy find_peaks (Prominence)**

Uses scipy's `find_peaks` with prominence metric:
- **Condition:** Peak prominence `≥ 5σ` AND minimum width `≥ 60 s`
- **Advantage:** Better separation of overlapping peaks in complex active regions
- **Implementation:** Stateless, easier for batch processing

### Method 3: **Dual-Window Detector** (Frozen Baseline)

Advanced statistical method combining two rolling windows:

1. **Short window (2–3 min):** Fast-moving foreground signal
2. **Long window (120–180 min):** Slow baseline
3. **Normalization:** Subtract baseline from foreground, divide by baseline std
4. **Frozen baseline:** Once a flare is suspected, baseline stops updating (prevents flare counts from poisoning the baseline estimate)
5. **Output:** z-score time series; alarm when `z > threshold` for `≥ persist_s` seconds

**Best Parameters (from grid search):**
```
long_window_min: 120.0
short_window_min: 3.0
freeze_threshold: 1.5
unfreeze_grace_sec: 120.0
signal_threshold: 1.5
min_duration_sec: 30.0
merge_gap_sec: 240.0
```

**Performance (on 2783 GT flares):**
- True Positives: **1625** | False Positives: **756**
- Precision: **68.3%** | Recall: **58.4%** | F1: **0.627**

### Method 4: **CUSUM (Cumulative Sum Control Chart)**

Accumulates evidence over time instead of checking single z-scores:

1. **Running total:** `g_pos` accumulates small consistent rises
2. **Resets:** When alarm triggers or no excess detected
3. **Advantage:** Naturally sensitive to slow gradual flares (small sustained excess over many minutes)
4. **Frozen baseline:** Uses same baseline-freezing strategy as dual-window
5. **Entry/Exit:** Fires when `g_pos ≥ decision_h`

**Best Parameters (from grid search):**
```
long_window_min: 120.0
short_smooth_sec: 10.0
freeze_z: 1.2
unfreeze_grace_sec: 60.0
drift_k: 0.75
decision_h: 15.0
min_duration_sec: 60.0
merge_gap_sec: 240.0
```

**Performance (on 2783 GT flares):**
- True Positives: **1722** | False Positives: **1424**
- Precision: **54.7%** | Recall: **61.9%** | F1: **0.581**

### Method 5: **Wavelet Detector** (Stationary Wavelet Transform)

Decomposes count rate into multiple time-scales simultaneously:

1. **SWT decomposition:** db4 wavelet, 6 levels
2. **Multi-scale detection:** Each wavelet scale captures different flare speeds
   - Fine scales: Fast impulsive flares
   - Coarse scales: Slow gradual flares
3. **Detection statistic:** Maximum z-score across selected scales
4. **Advantage:** Single threshold works for both impulsive and gradual flares
5. **Frozen baseline:** Applies to each wavelet scale independently

**Best Parameters (from grid search):**
```
wavelet: 'db4'
swt_levels: 6
use_levels: [3, 4, 5, 6]
long_window_min: 180.0
freeze_z: 1.5
unfreeze_grace_sec: 120.0
signal_threshold: 3.0
min_duration_sec: 60.0
merge_gap_sec: 240.0
```

**Performance (on 2783 GT flares):**
- True Positives: **1672** | False Positives: **1300**
- Precision: **56.3%** | Recall: **60.1%** | F1: **0.581**

### Method 6: **Probabilistic Detector** (Bayesian LLR + Welch's Test)

A causal, real-time-safe detector combining two signals:

1. **Signal A — Welch's t-test:**
   - Tests for significant difference between short foreground window and long background window
   - Background baseline **freezes** during suspected flare
   - Prevents rising flare counts from poisoning baseline statistics
   - Outputs: t-statistic and p-value

2. **Signal B — Bayesian Log-Likelihood-Ratio (LLR):**
   - Trained on TRAIN split using both flare and quiet seconds
   - Characterizes **both flare and quiet distributions** (not just flares)
   - Computes LLR of foreground/background intensity ratio
   - Outputs: log-odds favoring "flare" hypothesis

3. **Fusion:**
   - Combines both signals into posterior confidence score in [0, 1]
   - **Entry:** Confidence > (1-α) for ≥ persist_s seconds → flare state
   - **Exit:** Confidence < (1-β) for ≥ exit_persist_s seconds → quiet state
   - Hysteresis (β > α) prevents flickering at decision boundary

4. **Real-time deployment:**
   - `predict()` method takes only `.lc` file path
   - No ground-truth required; suitable for operational use

**Design Principle:**
> The innovation here is explicitly characterizing quiet-time behavior alongside flare behavior. Naive approaches (z-score only) characterize flares but never baseline statistics—this leads to instability when baseline changes. The LLR solves this by fitting both.

---

## Classification System

### GOES XRS Flux-Based Classification (`goes_classify.py`)

At each flare detection, the script fetches GOES-18 XRS 1–8 Å flux data and applies official thresholds:

| Class | Min Flux | Max Flux | Example |
|-------|----------|----------|---------|
| **X** | 1×10⁻⁴ W/m² | — | X2.5 = 2.5×10⁻⁴ W/m² |
| **M** | 1×10⁻⁵ W/m² | <1×10⁻⁴ | M6.0 = 6.0×10⁻⁵ W/m² |
| **C** | 1×10⁻⁶ W/m² | <1×10⁻⁵ | C3.2 = 3.2×10⁻⁶ W/m² |
| **B** | 1×10⁻⁷ W/m² | <1×10⁻⁶ | B5.1 = 5.1×10⁻⁷ W/m² |
| **A** | <1×10⁻⁷ W/m² | — | A8.0 = 8.0×10⁻⁸ W/m² |

**Subclass Number Calculation:** `subclass = flux / class_threshold`

```python
# Example:
flux = 6.0e-5  # W/m²
if flux >= 1e-4:
    class = "X"
    subclass = flux / 1e-4  # → X6.0
elif flux >= 1e-5:
    class = "M"
    subclass = flux / 1e-5  # → M6.0
# ... etc
```

#### Why GOES, Not SoLEXS Counts?

SoLEXS counts are in raw detector units (cts/s) that depend on:
- Aperture area
- Detector efficiency
- Energy-to-channel mapping (requires RMF + ARF calibration)

Without converting via RMF (Response Matrix File) + ARF (Ancillary Response File), counts cannot be converted to flux (W/m²). **GOES XRS provides calibrated, internationally standardized flux directly.**

#### GOES Data Retrieval (`goes_classify.py`)

```python
# Pseudocode
for each detected flare at peak_time:
    window = [peak_time - 5min, peak_time + 5min]
    goes_flux = sunpy.net.Fido(
        a.Time(window),
        a.Instrument('XRS')
    )
    max_flux = goes_flux[window].max()
    goes_class = classify_by_flux(max_flux)
```

---

## Results

### Neural Network Classifier (`nn_flareclassifier.py`)

**Architecture:** Multi-layer neural network trained for **30 epochs**

#### Training Set Performance

| Metric | No Flare | Flare |
|--------|----------|-------|
| Precision | 0.9842 | 0.6312 |
| Recall | 0.8439 | 0.5926 |
| F1-score | 0.9096 | 0.6113 |

**Overall Metrics:**
- Accuracy: **85.70%**
- Macro Avg F1: **0.7604**
- Weighted Avg F1: **0.8721**

**Confusion Matrix (Train):**
```
              Predicted No Flare | Predicted Flare
Actual No Flare      182,267     |     16,970
Actual Flare          16,398     |    195,396
```

#### Test Set Performance

| Metric | No Flare | Flare |
|--------|----------|-------|
| Precision | 0.9509 | 0.4119 |
| Recall | 0.9391 | 0.7199 |
| F1-score | 0.9450 | 0.5240 |

**Overall Metrics:**
- Accuracy: **82.79%**
- Macro Avg F1: **0.7345**
- Weighted Avg F1: **0.8415**

**Confusion Matrix (Test):**
```
              Predicted No Flare | Predicted Flare
Actual No Flare      113,908     |     25,403
Actual Flare           6,978     |     17,937
```

**Interpretation:**
- Excellent "No Flare" classification (precision ≈ 95%, recall ≈ 94%)
- "Flare" class more challenging: high recall (72%) but lower precision (41%)
- Trade-off reflects imbalanced detection problem where **missing a flare is costlier than false alarms**
- Suitable for screening applications where sensitivity is prioritized

---

### Random Forest & Gradient Boosting (`train_flare_models.ipynb`, `rfc.ipynb`)

Trained both classifiers on features extracted by `build_flare_dataset.py`:

- **Random Forest Classifier** — Ensemble of decision trees, interpretable feature importance
- **HistGradientBoostingClassifier** — Gradient boosting with histogram-based splits, faster training on large datasets
- Hyperparameters tuned via cross-validation on training split
- Separate evaluation on test split with per-GOES-class metrics

---

## Getting Started

### 1. Set Up Data Directory

```bash
mkdir -p lc_files pi_files data
```

### 2. Download and Extract SoLEXS Data

Download zipped SoLEXS files from [ISRO ISSDC Pradan](https://issdc.gov.in/), then:

```bash
python unzip_solex.py /path/to/downloads lc_files/
```

### 3. Inspect Raw Light Curves

```bash
python vis_lc.py lc_files/sample_file.lc
python vis_pi.py pi_files/sample_file.pi
```

### 4. Get Ground Truth Flare Data

```bash
python getFlares_data.py
# Outputs: data/hek_flares.csv
```

### 5. Build Feature Dataset

```bash
python build_flare_dataset.py
# Outputs: train/val/test datasets with extracted features
```

### 6. Run Detection & Classification

```bash
python goes_classify.py lc_files/sample_file.lc
# Generates: detection plot, GOES classifications, output CSV
```

### 7. Train ML Models

```bash
python nn_flareclassifier.py
# or open rfc.ipynb / train_flare_models.ipynb in Jupyter
```

---

## Usage Guide

### `unzip_solex.py`

**Purpose:** Extract SoLEXS data from ISRO downloads

```bash
python unzip_solex.py <source_directory> <target_directory>
```

**Behavior:**
- Recursively finds `.zip` files
- Extracts `.lc` and `.pi` FITS files
- Skips files with empty GTI extensions
- Preserves original filenames

---

### `clean_lcfiles.py`

**Purpose:** Remove problematic light curve files

```bash
python clean_lcfiles.py --dates 2024-01-15 2024-03-20 --directory lc_files/
```

**Use cases:**
- Remove files with instrument errors
- Discard dates with poor data quality
- Clean up corrupted downloads

---

### `vis_lc.py`

**Purpose:** Visualize individual light curve FITS files

```bash
python vis_lc.py lc_files/20240115_solex.lc --save output.png
```

**Output:** Time series plot showing:
- X-axis: Time (seconds from start)
- Y-axis: Count rate (cts/s)
- Flare signatures visible as count spikes

---

### `vis_pi.py`

**Purpose:** Visualize pulse invariant spectra

```bash
python vis_pi.py pi_files/20240115_solex.pi
```

**Output:** Energy spectrum showing X-ray photon distribution by energy channel

---

### `getFlares_data.py`

**Purpose:** Fetch HEK flare annotations for dates in your light curve dataset

```bash
python getFlares_data.py --start 2024-01-01 --end 2024-12-31
```

**Outputs:** `data/hek_flares.csv` with columns:
```
event_id, start_time, peak_time, end_time, goes_class, ar_longitude, ar_latitude
```

---

### `build_flare_dataset.py`

**Purpose:** Extract ML features from light curves and ground truth

```bash
python build_flare_dataset.py --lc-dir lc_files/ --hek-file data/hek_flares.csv
```

**Output Files:**
- `data/train_features.csv` — 70% of data, used for model training
- `data/val_features.csv` — 15% of data, used for hyperparameter tuning
- `data/test_features.csv` — 15% of data, final evaluation

**Feature Columns:**
```
time, counts, background, z_score, rise_slope, decay_slope, 
peak_counts, duration, energy, fluence, flare_class, has_flare
```

---

### `goes_classify.py`

**Purpose:** Classify detected flares using GOES XRS flux data

```bash
python goes_classify.py lc_files/20240115_solex.lc --save-plot output.png
```

**Workflow:**
1. Load SoLEXS light curve
2. Detect flares using threshold or find_peaks
3. Fetch GOES-18 XRS data via SunPy
4. Classify each detection by GOES flux
5. Generate combined and side-by-side plots
6. Output CSV with classifications

**Outputs:**
- `combined_plot.png` — Single figure with SoLEXS LC and GOES flux overlay
- `flare_1.png`, `flare_2.png`, ... — Individual flare plots (side-by-side)
- `goes_classifications.csv` — Detection results with GOES class

---

### `lc_flared.py`

**Purpose:** Overlay detected flares on light curves for visualization

```bash
python lc_flared.py lc_files/20240115_solex.lc --detections data/detections.csv
```

**Output:** Interactive or static plot showing:
- Light curve in black
- Detected flares highlighted in color
- Time intervals marked with shaded regions

---

### `solexs_flare_analysis.py`

**Purpose:** Run statistical analysis on light curve dataset

```bash
python solexs_flare_analysis.py --lc-dir lc_files/
```

**Outputs:**
- Summary statistics (mean, std, percentiles of count rates)
- Histogram of count rate distributions
- Auto-correlation analysis (flare timescale estimates)
- Power spectral density (noise characterization)

---

### `nn_flareclassifier.py`

**Purpose:** Train neural network for flare detection

```bash
python nn_flareclassifier.py --epochs 30 --batch-size 32 --features data/train_features.csv
```

**Architecture:**
```
Input (N features)
  ↓
Dense(128, relu) → Dropout(0.3)
  ↓
Dense(64, relu) → Dropout(0.3)
  ↓
Dense(32, relu) → Dropout(0.2)
  ↓
Dense(1, sigmoid) → Binary classification
```

**Output:** Model weights saved to `models/nn_classifier.h5`

---

### `rfc.ipynb` & `train_flare_models.ipynb`

**Purpose:** Interactive training of Random Forest and Gradient Boosting models

**Notebooks include:**
- Feature exploration and visualization
- Class imbalance analysis
- Hyperparameter grid search
- Cross-validation results
- Feature importance rankings
- Per-GOES-class performance breakdown
- Detection latency analysis

**Run in Jupyter:**
```bash
jupyter notebook rfc.ipynb
jupyter notebook train_flare_models.ipynb
```

---

## Model Performance

### Summary Comparison

| Detector | TP | FP | Precision | Recall | F1 |
|----------|----|----|-----------|--------|-----|
| **Dual-Window** | 1625 | 756 | 0.683 | 0.584 | 0.627 |
| **CUSUM** | 1722 | 1424 | 0.547 | 0.619 | 0.581 |
| **Wavelet** | 1672 | 1300 | 0.563 | 0.601 | 0.581 |
| **Neural Network (Test)** | 17,937 | 25,403 | 0.414 | 0.720 | 0.524 |

**Notes:**
- **Dual-Window** achieves best precision-recall balance
- **CUSUM** excels at detecting slow gradual flares
- **Wavelet** provides stable multi-scale sensitivity
- **Neural Network** shows high recall but lower precision (useful for screening)

### Recommendations

- **For high-confidence catalog:** Use dual-window (best F1)
- **For sensitive detection (minimize misses):** Use CUSUM + Wavelet ensemble
- **For real-time deployment:** Use probabilistic detector (causal, no future data needed)
- **For detailed feature analysis:** Use Random Forest (interpretable importance)

---

## References & Terminology

- **HEK:** Heliophysics Event Knowledgebase — ground truth solar flare catalog
- **GOES XRS:** Geostationary Operational Environmental Satellite X-Ray Sensor (1–8 Å channel)
- **SoLEXS:** Soft Lunar X-ray Spectrometer on Chandrayaan-2 Orbiter
- **GTI:** Good Time Interval — periods of valid instrument operation
- **RMF/ARF:** Calibration matrices (Response Matrix File / Ancillary Response File)
- **LLR:** Log-Likelihood Ratio — Bayesian evidence metric
- **SWT:** Stationary Wavelet Transform — multi-scale decomposition
- **CUSUM:** Cumulative Sum Control Chart — sequential change detection
- **F1-score:** Harmonic mean of precision and recall

---

## Future Work

- [ ] Ensemble voting from multiple detectors
- [ ] Flare forecasting (predict tomorrow's events using yesterday's data)
- [ ] Spectral hardness ratio analysis
- [ ] Automated pipeline with real-time alerts
- [ ] Compare against official GOES event lists
- [ ] Deploy probabilistic detector in production environment

---

## License

[Specify your license here]

## Contact

For questions or contributions, contact: [Your contact information]

---

**Last updated:** July 2026  
**Status:** Active development
