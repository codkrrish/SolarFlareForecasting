<div align="center">

# 🌞 Solar Flare Detection & Forecasting

### **Real-Time Solar Flare Detection and Classification using Statistical Signal Processing and Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success)]()



### **A comprehensive framework for real-time solar flare detection and standardized GOES classification using X-ray light curve observations from SoLEXS onboard ISRO's Aditya-L1 mission.**

</div>

---

# 📑 Table of Contents

- 🌍 Why This Project?
- ✨ Key Features
- 🔄 Current Pipeline
- 🛠️ Technology Stack
- 📂 Repository Structure
- 📊 Dataset
- 🚀 Getting Started
- 📡 Statistical Detection Methods
- 🧠 Machine Learning Models
- 🛰️ GOES Classification
- 📈 Results & Evaluation
- 💻 Usage
- 🗺️ Future Work
- 🤝 Contributing
- 📜 References
- 📄 License

---

# 🌍 Why This Project?

Solar flares are sudden bursts of electromagnetic radiation released due to magnetic reconnection on the Sun. These events are among the most energetic phenomena in our solar system and can significantly affect modern technological infrastructure.

Solar flares can

- 🛰️ Disrupt satellite operations
- 📡 Affect GPS and radio communication
- ⚡ Trigger geomagnetic storms
- ✈️ Increase radiation exposure during high-altitude flights
- 👨‍🚀 Endanger astronauts and spacecraft electronics

Reliable and automated flare detection is therefore essential for space weather forecasting and early warning systems.

This project develops a complete end-to-end framework for detecting solar flares from **SoLEXS X-ray Light Curve (LC)** observations and assigning standardized **GOES flare classes** using multiple statistical methods and machine learning models.

---

# ✨ Key Features

- ✅ Real-time compatible statistical detection
- ✅ Multiple independent flare detectors
- ✅ Bayesian probabilistic detection framework (Work in Progress)
- ✅ Random Forest classifier
- ✅ HistGradientBoosting classifier
- ✅ Neural Network classifier
- ✅ HEK ground-truth validation
- ✅ GOES XRS standardized flare classification
- ✅ Detection latency evaluation
- ✅ Precision, Recall and F1-score analysis
- ✅ Interactive visualization tools
- ✅ Modular and extensible pipeline

---

# 🔄 Current Pipeline

> **Current research workflow (not the final deployment architecture)**

```text
SoLEXS Light Curve (.lc)
            │
            ▼
    Data Cleaning
            │
            ▼
 Statistical Detection
(Welch's t-test + Bayesian LLR)
            │
            ▼
 Feature Extraction
            │
            ▼
 Random Forest Classifier
            │
            ▼
 GOES Class Assignment
            │
            ▼
 Visualization & Evaluation
```

---

# 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming | Python |
| Data Processing | NumPy • Pandas |
| Scientific Computing | SciPy |
| Visualization | Matplotlib |
| FITS Processing | Astropy |
| Solar Physics | SunPy |
| Machine Learning | Scikit-Learn |
| Deep Learning | TensorFlow / Keras |
| Statistical Detection | Welch's t-test • CUSUM • Stationary Wavelet Transform |
| Probabilistic Methods | Bayesian Log-Likelihood Ratio |
| ML Models | Random Forest • HistGradientBoosting • Neural Network |
| Dataset | SoLEXS (Aditya-L1), HEK, GOES XRS |
| Version Control | Git • GitHub |

---

## ⚙️ Technologies Used

<p align="center">

<img src="https://skillicons.dev/icons?i=python,tensorflow,sklearn,git,github"/>

</p>

<p align="center">

<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Astropy-FF6600?style=for-the-badge"/>
<img src="https://img.shields.io/badge/SunPy-F9A602?style=for-the-badge"/>

</p>

<p align="center">

<img src="https://img.shields.io/badge/Random%20Forest-228B22?style=for-the-badge"/>
<img src="https://img.shields.io/badge/HistGradientBoosting-1E88E5?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Neural%20Network-8E44AD?style=for-the-badge"/>
<img src="https://img.shields.io/badge/CUSUM-E91E63?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Welch's%20t--Test-009688?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Bayesian%20LLR-FF5722?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Stationary%20Wavelet%20Transform-3F51B5?style=for-the-badge"/>

</p>

---
## 📂 Repository Structure

```text
SolarFlareForecasting/
│# Data Processing
├── unzip_solex.py
├── clean_lcfiles.py
├── vis_lc.py
├── vis_pi.py
└── solexs_flare_analysis.py
│
├#Data Preparation
├── build_flare_dataset.py
├── getFlares_data.py
└── goes_classify.py
│
├# Machine Learning
├── train_flare_models.ipynb
├── t-BLL_FlareDetector.py
└── nn_flareclassifier.py
│
├# Visualization
└── lc_flared.py
│
├─images/
│
└── README.md
```

---

# 📊 Dataset

This framework integrates multiple publicly available datasets to perform robust solar flare detection and classification.

## ☀️ 1. SoLEXS (Aditya-L1)

The primary dataset consists of X-ray Light Curve (LC) observations collected by the **Solar Low Energy X-ray Spectrometer (SoLEXS)** onboard **ISRO's Aditya-L1 mission**.

Used files:

- `.lc` Light Curve FITS
- `.pi` Pulse Invariant FITS

These provide the raw X-ray photon counts used throughout the pipeline.

---

## 🌞 2. HEK (Heliophysics Event Knowledgebase)

HEK is used **only as ground truth** for evaluation.

Information extracted:

- Event ID
- Start Time
- Peak Time
- End Time
- GOES Class

This allows automated matching between detected events and officially recorded solar flares.

---

## 🛰️ 3. GOES XRS

GOES X-Ray Sensor observations are used **only after flare detection** to assign the official flare class.

Classes include

| Class | Flux Range |
|---------|----------------|
| A | <10⁻⁷ W/m² |
| B | 10⁻⁷ – 10⁻⁶ |
| C | 10⁻⁶ – 10⁻⁵ |
| M | 10⁻⁵ – 10⁻⁴ |
| X | >10⁻⁴ |

---

# 🚀 Getting Started

## Requirements

- Python 3.10+
- Git

Install dependencies

```bash
pip install numpy pandas scipy matplotlib scikit-learn
pip install astropy sunpy sunpy[net]
pip install tensorflow
```

---

## Clone Repository

```bash
git clone https://github.com/<username>/SolarFlareForecasting.git

cd SolarFlareForecasting
```

Replace with your actual repository link.

---

## Prepare Dataset

```bash
mkdir lc_files
mkdir pi_files
mkdir data
```

Extract downloaded SoLEXS files

```bash
python unzip_solex.py <download_folder> lc_files/
```

---

## Download Ground Truth

```bash
python getFlares_data.py
```

This generates

```
data/hek_flares.csv
```

---

## Build Dataset

```bash
python build_flare_dataset.py
```

Outputs

```
train_features.csv

validation_features.csv

test_features.csv
```

---

## Train Models

Random Forest & Gradient Boosting

```bash
jupyter notebook train_flare_models.ipynb
```

Neural Network

```bash
python nn_flareclassifier.py
```

---

## Run Detection

```bash
python goes_classify.py lc_files/sample.lc
```

Outputs

- GOES Class
- Detection plots
- CSV report

---

# 📡 Statistical Detection Methods

The framework implements multiple statistical detectors. Each detector has different strengths and complements the others.

---

## 1️⃣ Threshold Detector

A simple baseline approach.

**Rule**

```
counts > background + 5σ
```

for at least

```
60 seconds
```

### Advantages

- Extremely fast
- Easy to interpret

### Limitation

Sensitive to background variation.

---

## 2️⃣ Peak Detection (SciPy)

Uses

```
scipy.signal.find_peaks()
```

Conditions

- Prominence ≥ 5σ
- Minimum width ≥ 60 s

Suitable for separating nearby flare peaks.

---

## 3️⃣ Dual Window Detector

Maintains

- Short foreground window
- Long background window

When flare activity begins,

the background estimate is frozen,

preventing contamination.

Performance

| Metric | Value |
|---------|------|
| Precision | **68.3%** |
| Recall | 58.4% |
| F1 | **0.627** |

---

## 4️⃣ CUSUM Detector

Detects gradual changes by accumulating small increases over time.

Best suited for

- Weak flares
- Slowly rising events

Performance

| Metric | Value |
|---------|------|
| Precision | 54.7% |
| Recall | **61.9%** |
| F1 | 0.581 |

# 📡 Advanced Statistical Detection Methods

## 5️⃣ Wavelet Detector (Stationary Wavelet Transform)

Instead of observing the signal at only one scale, the framework decomposes the light curve into multiple frequency bands using the **Stationary Wavelet Transform (SWT)**.

### Methodology

- Wavelet: **Daubechies-4 (db4)**
- SWT Levels: **6**
- Detection Levels: **3–6**
- Frozen baseline strategy
- Multi-scale z-score computation

### Why Wavelets?

Different solar flares evolve differently.

- Fine scales capture **impulsive flares**
- Coarse scales capture **slow gradual flares**

This enables detection of a wider variety of flare morphologies using a single statistical framework.

### Best Parameters

| Parameter | Value |
|-----------|-------|
| Wavelet | db4 |
| Levels | 6 |
| Detection Levels | 3,4,5,6 |
| Background Window | 180 min |
| Freeze Threshold | 1.5 |
| Signal Threshold | 3 |
| Minimum Duration | 60 s |

### Performance

| Metric | Value |
|---------|------|
| True Positives | 1672 |
| False Positives | 1300 |
| Precision | **56.3%** |
| Recall | **60.1%** |
| F1 Score | **0.581** |

---

## 6️⃣ Probabilistic Detector (Welch's t-test + Bayesian LLR)

This is the primary research direction of the project and the proposed detector for deployment.

Unlike traditional threshold-based methods, it combines statistical hypothesis testing with probabilistic inference.

### Stage 1 — Welch's t-test

A foreground window is continuously compared against the background window.

The detector evaluates whether the current signal differs significantly from the historical background.

Features:

- Frozen background estimation
- Robust against changing baselines
- No future information required
- Suitable for streaming data

---

### Stage 2 — Bayesian Log-Likelihood Ratio (LLR)

The Bayesian model learns the distributions of

- Quiet solar activity
- Flare activity

It computes the probability that the current signal belongs to either class.

Instead of relying only on z-scores,

it estimates

```
P(Flare | Observation)
```

using Bayesian inference.

---

### Decision Logic

The detector combines

- Welch's t-test
- Bayesian confidence

to produce a final flare probability.

```
Light Curve
      │
      ▼
Foreground vs Background
      │
      ▼
Welch's t-test
      │
      ▼
Bayesian Log-Likelihood Ratio
      │
      ▼
Posterior Probability
      │
      ▼
Flare / Quiet Decision
```

### Advantages

✅ Real-time compatible

✅ Causal (no future samples)

✅ Robust to background drift

✅ Lower false alarms

✅ Probabilistic confidence score

---

# 🧠 Machine Learning Models

After candidate flare intervals are detected,

hand-crafted features are extracted and passed to machine learning models for final prediction.

---

## 🌳 Random Forest Classifier

The primary classical machine learning model.

### Characteristics

- Ensemble of Decision Trees
- Bootstrap aggregation
- Feature importance estimation
- Robust against noisy features
- Handles nonlinear relationships

### Why Random Forest?

- Highly interpretable
- Strong baseline performance
- Minimal preprocessing
- Resistant to overfitting

---

## 🚀 HistGradientBoosting Classifier

Histogram-based Gradient Boosting optimized for large datasets.

### Advantages

- Faster than traditional Gradient Boosting
- Lower memory usage
- Excellent nonlinear modeling
- High scalability

---

## 🧠 Neural Network

A Multi-Layer Perceptron (MLP) is implemented for high-recall flare detection.

### Architecture

```
Input Features
      │
      ▼
Dense (128) + ReLU
      │
Dropout (0.3)
      │
      ▼
Dense (64) + ReLU
      │
Dropout (0.3)
      │
      ▼
Dense (32) + ReLU
      │
Dropout (0.2)
      │
      ▼
Sigmoid Output
```

Training

- Epochs: 30
- Optimizer: Adam
- Binary Cross Entropy Loss

The neural network emphasizes **high recall**, making it useful as a screening model where missing a flare is more costly than generating additional false positives.

---

# 🛰️ GOES Classification

Once a flare has been detected,

its official classification is assigned using **GOES X-Ray Sensor (XRS)** observations.

Classification follows the internationally accepted GOES standard.

| Class | Peak Flux |
|---------|----------------|
| A | < 10⁻⁷ W/m² |
| B | 10⁻⁷ – 10⁻⁶ |
| C | 10⁻⁶ – 10⁻⁵ |
| M | 10⁻⁵ – 10⁻⁴ |
| X | > 10⁻⁴ |

Example

```
Flux = 6.5 × 10⁻⁵

↓

GOES Class = M6.5
```

The GOES class is attached to every detected flare to provide standardized reporting and comparison with existing solar event catalogs.

# 📈 Results & Evaluation

The framework was evaluated on **2,783 ground-truth solar flare events** obtained from the **Heliophysics Event Knowledgebase (HEK)**.

## 📊 Statistical Detector Performance

| Detector | True Positives | False Positives | Precision | Recall | F1 Score |
|------------|---------------:|---------------:|----------:|-------:|---------:|
| 🥇 Dual-Window | 1625 | **756** | **68.3%** | 58.4% | **0.627** |
| 📈 CUSUM | **1722** | 1424 | 54.7% | **61.9%** | 0.581 |
| 🌊 Wavelet | 1672 | 1300 | 56.3% | 60.1% | 0.581 |

---

## 🧠 Neural Network Performance

### Training Performance

| Metric | No Flare | Flare |
|---------|----------|--------|
| Precision | 98.42% | 63.12% |
| Recall | 84.39% | 59.26% |
| F1 Score | 90.96% | 61.13% |

**Overall Accuracy:** **85.70%**

---

### Test Performance

| Metric | No Flare | Flare |
|---------|----------|--------|
| Precision | 95.09% | 41.19% |
| Recall | 93.91% | **71.99%** |
| F1 Score | 94.50% | 52.40% |

**Overall Accuracy:** **82.79%**

---

### 📌 Key Observations

- ✅ **Dual-Window Detector** achieved the highest overall F1-score among the statistical detectors.
- ✅ **CUSUM** provided the highest recall, making it effective for detecting slowly evolving flares.
- ✅ **Wavelet Detector** demonstrated stable performance across multiple flare timescales.
- ✅ **Neural Network** achieved high recall, making it suitable for early-warning and screening applications.

---

# 📊 Visualizations

Replace these placeholders with actual project outputs.

## Light Curve Detection

<p align="center">
<img src="images\lc_plot.jpeg" width="80%">
</p>

---

## GOES Classification

<p align="center">
<img src="images\ground_truth.jpeg" width="80%">
</p>

---


# 💻 Usage

## Visualize Light Curve

```bash
python vis_lc.py lc_files/sample.lc
```

---

## Visualize PI Spectrum

```bash
python vis_pi.py pi_files/sample.pi
```

---

## Download HEK Events

```bash
python getFlares_data.py
```

---

## Build Training Dataset

```bash
python build_flare_dataset.py
```

---

## Run Detection Pipeline

```bash
python goes_classify.py lc_files/sample.lc
```

---

## Train Neural Network

```bash
python nn_flareclassifier.py
```

---

## Train Random Forest & Gradient Boosting

```bash
jupyter notebook train_flare_models.ipynb
```

---

## Run Random Forest Notebook

```bash
jupyter notebook rfc.ipynb
```

---

# 📊 Current Research Status

| Component | Status |
|-----------|--------|
| Data Extraction | ✅ Completed |
| Dataset Preparation | ✅ Completed |
| Feature Engineering | ✅ Completed |
| Statistical Detectors | ✅ Completed |
| Random Forest | ✅ Completed |
| HistGradientBoosting | ✅ Completed |
| Neural Network | ✅ Completed |
| GOES Classification | ✅ Completed |
| Evaluation Pipeline | ✅ Completed |
| Probabilistic Detector | 🚧 In Progress |
| Real-time Deployment | 🚧 Planned |

---

# 🎯 Highlights

- ✅ Real-time compatible framework
- ✅ Multiple statistical detectors
- ✅ Random Forest baseline
- ✅ Gradient Boosting implementation
- ✅ Neural Network classifier
- ✅ HEK validation
- ✅ GOES standardized classification
- ✅ Interactive visualization
- ✅ End-to-end machine learning pipeline
- ✅ Designed for future real-time deployment

---
# 🗺️ Future Work

The project is under active development, with several planned improvements aimed at enhancing detection accuracy, robustness, and real-time deployment capabilities.

## 🚀 Planned Improvements

- [ ] Complete the **Welch's t-test + Bayesian Log-Likelihood Ratio (LLR)** detector and integrate it into the main pipeline.
- [ ] Develop an ensemble framework combining statistical detectors and machine learning models for improved robustness.
- [ ] Explore advanced time-series architectures such as **iTransformer**, **PatchTST**, and **Temporal Fusion Transformer (TFT)**.
- [ ] Improve feature engineering using additional temporal and statistical descriptors.
- [ ] Optimize detection latency for real-time space weather monitoring.
- [ ] Develop a web-based dashboard for visualization and monitoring.
- [ ] Build a REST API for automated flare detection and classification.
- [ ] Benchmark the framework against additional public solar flare datasets.
- [ ] Improve explainability using feature importance and model interpretation techniques.
- [ ] Containerize the project using Docker for easier deployment.

---

# 🤝 Contributing

Contributions are welcome!

If you have ideas to improve the project, feel free to open an issue or submit a pull request.

### Steps to contribute

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature/YourFeature
```

3. Commit your changes

```bash
git commit -m "Add YourFeature"
```

4. Push to GitHub

```bash
git push origin feature/YourFeature
```

5. Open a Pull Request 🚀

---

# 📜 References

- **ISRO Aditya-L1 Mission**
- **SoLEXS (Solar Low Energy X-ray Spectrometer) Documentation**
- **HEK (Heliophysics Event Knowledgebase)**
- **GOES X-Ray Sensor (XRS) Documentation**
- **Welch, B. L. (1947). The Generalization of Student's Problem when Several Different Population Variances are Involved.**
- **Page, E. S. (1954). Continuous Inspection Schemes (CUSUM).**
- **Mallat, S. (1999). A Wavelet Tour of Signal Processing.**
- **Breiman, L. (2001). Random Forests. Machine Learning.**
- **Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine.**

---

# 📄 License

This project is released under the **MIT License**.

See the **LICENSE** file for complete details.

---

# 📬 Contact

**Project:** Solar Flare Detection & Forecasting

**Maintainers:** Krrish Swarnkar, Ram Kumar, Gargi Pareek, Mahitha JV
                



---

# ⭐ Support the Project

If you found this project useful,

please consider giving it a ⭐ on GitHub.

It helps the project reach more researchers, developers, and students interested in **Space Weather**, **Machine Learning**, and **Scientific AI**.

---

<div align="center">

## 🌞 Built with ❤️ for Space Weather Research

### If you like this project, don't forget to ⭐ star the repository!

**Happy Coding! 🚀**

</div>