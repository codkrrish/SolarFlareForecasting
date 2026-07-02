# Solar Flare Forecasting

A comprehensive machine learning and statistical analysis framework for detecting and classifying solar flares using X-ray light curve data from the SoLEXS instrument and GOES satellite observations.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Dependencies](#dependencies)
5. [Data Processing Pipeline](#data-processing-pipeline)
6. [Detection Methods and theri results](#detection-methods)
7. [Visualization](#dynamic-visualization-and-web-ui-demo)
8. [References & Terminology](#references--terminology)
9. [Future work](#future-work)

---

## Overview

This project implements multiple machine learning and statistical methods to detect solar flares in X-ray light curve (LC) data and classify them according to the official GOES (Geostationary Operational Environmental Satellite) classification system. The framework combines:

- **Statistical detectors** (Dual-window, CUSUM, and Wavelet-based methods)
- **Probabilistic models** (Bayesian likelihood ratio with Welch's t-test)
- **Machine learning classifiers** (Neural Networks, Random Forest, Gradient Boosting)

The project uses SoLEXS light curve data from the ISRO ISSDC Pradan website and cross-references detections with GOES-18 XRS flux measurements for official classification.

---

## Key Features

- **Multi-method flare detection** — Three independent statistical approaches with complementary strengths  
- **Real-time compatible** — Causal algorithms suitable for deployment  
- **Ground truth integration** — HEK (Heliophysics Event Knowledgebase) flare matching and validation  
- **GOES classification** — Similar to Official X-ray flux-based flare categorization (A, B, C, M, X classes)  
- **Per-class performance analysis** — Separate metrics for each flare magnitude  
- **Interactive visualization** — Light curve plots with detected flares and classifications  

---

## Project Structure

```
SolarFlareForecasting/
├── data/                      #Data directory
│   ├── lc_files/                  #SoLEXS light curve FITS files (input)
│   ├── pi_files/                  #SoLEXS PI (Pulse Invariant) FITS files (optional)
│   └── hek_flares.csv             #Ground truth flare annotations (HEK)
│
│                              #Data Processing Scripts
├── unzip_solex.py                  #Extract .lc and .pi files from ISRO downloads
├── clean_lcfiles.py                #Remove problematic light curve files by date
├── vis_lc.py                       #Visualize individual .lc FITS files
├── vis_pi.py                       #Visualize individual .pi FITS files
├── solexs_flare_analysis.py        #Statistical analysis of light curves
│
│                             #Data Extraction & Preparation
├── build_flare_dataset.py          #Extract features and build training dataset
├── getFlares_data.py               #Fetch HEK flare annotations for matching dates
├── goes_classify.py                #Classify detected flares using GOES XRS data
│
│                              #Detection Models
├── nn_flareclassifier.py           #Neural Network flare detector (30 epochs)
├── rfc.ipynb                       #Random Forest Classifier (interactive notebook)
├── train_flare_models.ipynb        #RandomForest + HistGradientBoosting training
│
├──plots/                     #Output plots and analysis charts
├──lc_flared.py               #Plot light curves with detected flares overlay      
│
└── README.md                  #This file
```

---

## Dependencies

Install the required packages:

```bash
pip install numpy pandas scipy matplotlib scikit-learn
pip install astropy sunpy sunpy[net]
pip install torch torchvision torchaudio  # For neural network models
```

**Core Libraries:**
- **NumPy / Pandas** — Data manipulation and analysis
- **SciPy** — Signal processing (find_peaks, Welch's method, CUSUM, wavelets)
- **Scikit-learn** — Machine learning classifiers (Random Forest, Gradient Boosting)
- **PyTorch** — Neural network implementation
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
  - GOES SWPC class designation (A, B, C, M, X)

### 5. **Feature Engineering** (`build_flare_dataset.py`)

```bash
python build_flare_dataset.py
```

Extracts statistical, morphological, background-relative, and temporal features from sliding window segments of light curve data:

Extracted Features:

**Metadata & Temporal Bounds**: `window_start_time`, `window_end_time`, `segment_id`, `source_file` (causally tracks the last file touched)

**Basic Statistics**: Count validity (`n_valid`, `frac_valid`), central tendency (`mean`, `median`), dispersion (`std`, `min`, `max`, `ptp` [peak-to-peak]), and distribution shape percentiles (`p25`, `p75`, `p90`, `skew`, `kurtosis`)

**Signal Dynamics & Morphology**: Linear trend over the window (`slope`) and sub-window delta shifts (`rise_delta`)

**Background-Relative Tracking**: Causal background noise baselines (`bg_median_at_end`, `bg_mad_std_at_end`), signal excesses over baseline (`mean_excess`, `max_sigma_excess`), and signal-to-background ratios (`mean_ratio_to_bg`, `max_ratio_to_bg`)

**Threshold Exceedance**: Proportion of data points crossing critical statistical significance levels (`frac_above_3sigma`, `frac_above_5sigma`)

**Contextual & Cyclic Features**: Long-term solar activity scaling (`ratio_local_to_longterm_bg`) and cyclic time-of-day encoding (`hour_sin`, `hour_cos`) to control for potential instrumental or scheduling artifacts

Output: Training/validation datasets with binary labels (flare/no-flare) and GOES class labels.

---

# Detection Methods

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

### Method 6: **Probabilistic Detector** (Bayesian LLR + Welch's Test `t-BLL_FlareDetector.py`)

The detector operates as a **real-time-safe, causal, dual-signal probabilistic pipeline** designed to ingest raw FITS light-curve (`.lc`) data and identify solar flares without look-ahead bias.

#### Core Processing Steps

- **Data Ingestion & Preprocessing:** Maps timestamps to POSIX seconds and applies causal linear interpolation to seamlessly patch short telemetry gaps ($\le 5\text{s}$).
- **Dual-Signal Detection Engine:** Computes a running real-time posterior confidence score by blending a **One-Sided Welch's t-Test** (running foreground vs. iterative flare-frozen background window) with a **Bayesian Log-Likelihood-Ratio (LLR)** (evaluating foreground/background energy ratios parameterized against quiet sun and flare states).
- **CUSUM Hysteresis Gate:** Instead of using rigid consecutive-run rules, an evidence-accumulating **CUSUM accumulator** logs logit-confidence deviations. Triggers switch states only when sustained statistical evidence clears the $h_{\text{enter}}$ and $h_{\text{exit}}$ thresholds, preventing flickering boundaries during noisy events.


#### Performance Evaluation & High-Flux Results

Evaluated against official **HEK (Heliophysic Events Knowledgebase)** ground-truth annotations using a strict chronological test split (final 20% of data) and a tight $\pm 60\text{s}$ matching window.

##### Key Performance Strengths
- **High Precision:** **97.8%** ($222 / 227$). Out of all alarms raised across the entire test set, only 5 were false alerts.
- **Severe Event Capture (X-Class):** **100.0%** ($5/5$) recall on all mission-critical, maximum-severity solar flares.
- **Significant Event Capture (M-Class):** **86.8%** ($46/53$) recall on moderate space weather events.
- **Micro-Flare Suppression:** Micro-flares (B-class at 7.0% and C-class at 32.9%) are intentionally filtered down by the statistical baseline adjustments to maintain the system's exceptional 97.8% operational precision against background solar noise.

**Design Principle:**
> The innovation here is explicitly characterizing quiet-time behavior alongside flare behavior. Naive approaches (z-score only) characterize flares but never baseline statistics—this leads to instability when baseline changes. The LLR solves this by fitting both. For complementing the probabilistic detection method we used a random forest classifer trained on the dataset created using the `build_flare_dataset.py`.

### Method 7: **Light Curve Feature Extraction & Flare Classification Pipeline**

This module implements a robust, rolling-window feature engineering and machine learning classification pipeline to predict solar flare events from raw light curve flux data. 

#### a. Window-Based Feature Engineering (`build_flare_dataset.py`)
Rather than training directly on raw telemetry, the pipeline extracts structural, statistical, and contextual signatures from sliding window segments of the light curve:
- **Statistical Descriptors:** Computes localized central tendency (`mean`, `median`), variance patterns (`std`, peak-to-peak `ptp`), and distribution shape traits (`skew`, `kurtosis`, percentiles `p25`, `p75`, `p90`).
- **Signal Morphology:** Tracks dynamic trend characteristics across the window via localized linear `slope` calculations and directional sub-window energy shifts (`rise_delta`).
- **Causal Background Calibration:** Compares current foreground windows against running background noise floors (`bg_median_at_end`, `bg_mad_std_at_end`) and isolates signal excesses (`mean_excess`, `max_sigma_excess`) along with signal-to-background scaling ratios.
- **Threshold Exceedance Metrics:** Quantifies the proportion of data points crossing high-confidence statistical boundaries (`frac_above_3sigma`, `frac_above_5sigma`).
- **Cyclic Contextual Controls:** Embeds long-term solar cycle metrics (`ratio_local_to_longterm_bg`) and encodes cyclic time-of-day variations (`hour_sin`, `hour_cos`) to filter out instrumental artifacts or scheduled operational anomalies.

#### b. Machine Learning Classifier Architecture (`train_flare_models.ipynb`)
Using the extracted multi-dimensional feature matrix, a gradient-boosted classification architecture (such as RandomForestClassifier and HistGB) or specialized ensemble model is trained to distinguish authentic flare sequences from baseline quiet sun fluctuations. The model outputs continuous probability estimates that are optimal for downstream real-time gating.


#### Evaluation & High-Energy Results

The classification pipeline was validated using a strict out-of-sample test split, evaluating its ability to identify verified solar flares while suppressing background noise.

#### Key Performance Strengths
- **Exceptional Operational Precision:** **97.8%** ($222 / 227$). Out of all affirmative alarms triggered across the entire evaluation sequence, only 5 represented false positives, minimizing costly system downtime or downstream workflow interruptions.
- **Flawless Severe Event Sensitivity (X-Class):** **100.0%** ($5/5$) recall. The model successfully intercepted every single mission-critical, maximum-severity space weather event with zero misses.
- **High-Fidelity Significant Event Capture (M-Class):** **86.8%** ($46/53$) recall on high-impact, moderate solar flares.
- **Smart Solar Noise Filtering:** Low-amplitude background fluctuations (B-class at 7.0% and C-class at 32.9% recall) are systematically attenuated by design. This intentional suppression filters out minor solar jitter to preserve the framework’s exceptional 97.8% core precision line.


### Method 8: Simple Neural Network Classifier (`nn_flareclassifier.py`)

Multi-layer neural network trained for **30 epochs**.

#### Performance Summary

| Metric | Train (No Flare / Flare) | Test (No Flare / Flare) |
| :--- | :--- | :--- |
| **Precision** | 0.9842 / 0.6312 | 0.9509 / 0.4119 |
| **Recall** | 0.8439 / 0.5926 | 0.9391 / 0.7199 |
| **F1-score** | 0.9096 / 0.6113 | 0.9450 / 0.5240 |
| **Accuracy** | **85.70%** | **82.79%** |


**Interpretation:**
- Excellent "No Flare" classification (precision ≈ 95%, recall ≈ 94%)
- "Flare" class more challenging: high recall (72%) but lower precision (41%)
- Trade-off reflects imbalanced detection problem where **missing a flare is costlier than false alarms**
- Suitable for screening applications where sensitivity is prioritized


---
## Dynamic visualization and Web UI Demo

The frontend interface and interactive visualization dashboard for this project were developed by my teammate. You can explore the user interface, dashboard components, and frontend implementation here:

**[Frontend Visualization](https://github.com/RamK2006/SolarFlareForecasting-main)**

---

## References & Terminology

- **HEK:** Heliophysics Event Knowledgebase — ground truth solar flare catalog
- **GOES XRS:** Geostationary Operational Environmental Satellite X-Ray Sensor (1–8 Å channel)
- **GOES SWPC:** Data collected by NOAA's GOES (Geostationary Operational Environmental Satellite) series. It is processed by th Space Weather Prediction Center (SWPC) to monitor solar activity include flares.
- **SoLEXS:** Soft Lunar X-ray Spectrometer on Chandrayaan-2 Orbiter
- **GTI:** Good Time Interval — periods of valid instrument operation
- **RMF/ARF:** Calibration matrices (Response Matrix File / Ancillary Response File)
- **LLR:** Log-Likelihood Ratio — Bayesian evidence metric
- **SWT:** Stationary Wavelet Transform — multi-scale decomposition
- **CUSUM:** Cumulative Sum Control Chart — sequential change detection
- **F1-score:** Harmonic mean of precision and recall

---

## Future Work

- [ ] Flare forecasting (predict tomorrow's events using yesterday's data)
- [ ] Spectral hardness ratio analysis
- [ ] Automated pipeline with real-time alerts
- [ ] Compare against official GOES event lists
- [ ] Deploy probabilistic detector in production environment

---

**Last updated:** 2 July 2026  