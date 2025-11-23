# Anomaly Detection in High-Energy Physics using Deep Autoencoders

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 🔬 Project Overview

This project applies **Deep Learning** techniques to a fundamental problem in High-Energy Physics (HEP): detecting anomalous hadronic jets produced at the **Large Hadron Collider (LHC)**.

At extreme energies, massive particles decay into hadronic jets that are highly collimated. Distinguishing whether a jet substructure originates from a known Standard Model particle (background) or a potential new massive particle (signal/anomaly) is a complex task. 

We implemented a **Convolutional Autoencoder (CNN-AE)** to perform unsupervised anomaly detection, training the model to reconstruct "normal" jets and identifying anomalies based on reconstruction error and latent space analysis.

## 🎯 Key Objectives

* **Unsupervised Learning:** Train a model exclusively on Standard Model background data (QCD jets).
* **Architecture:** Design a Convolutional Autoencoder capable of compressing 2D calorimeter images into a dense latent representation.
* **Anomaly Scoring:** Compare different metrics for outlier detection:
    * **MSE (Reconstruction Loss):** $L = ||x - \hat{x}||^2$
    * **Mahalanobis Distance:** Calculated in the latent space.
* **Latent Space Analysis:** Use **PCA** and **Gaussian Mixture Models (GMM)** to interpret how the network organizes physical features.

## 🛠️ Methodology & Tech Stack

* **Frameworks:** Python, PyTorch, Scikit-Learn, NumPy, Matplotlib.
* **Data Input:** 2D images ($100 \times 100$ pixels) representing energy deposits in calorimeter cells.
* **Model Architecture:** * *Encoder:* 3 Convolutional blocks + MaxPool + Linear projection to $\mathbb{R}^{10}$.
    * *Decoder:* Mirrored structure with ConvTranspose layers to reconstruct the input.
* **Optimization:** Adam Optimizer, MSE Loss.

## 📊 Results

### 1. Reconstruction Loss Analysis
The model successfully learned to reconstruct background jets with low error, while failing to reconstruct anomalous jets (signals). This discrepancy allows for effective signal-background separation.

> **Note:** [Insert your Histogram of Loss plot here]

### 2. Latent Space Visualization
By projecting the 10-dimensional latent space onto 2D using **PCA**, we observed distinct clustering behaviors between the background and the anomalous signals, validating the model's feature extraction capabilities.

> **Note:** [Insert your PCA scatter plot here]

### 3. Performance
* **False Positive Rate (FPR):** Maintained $\le 10\%$ on background data.
* **Signal Discrimination:** We confirmed the physical hypothesis that higher mass particles ($f_{high}$) result in higher reconstruction errors compared to lower mass anomalies ($f_{low}$), i.e., $f_{high} > f_{low}$.

## 📂 Data Availability

The dataset used in this project consists of simulated LHC events provided by **Prof. Stefano Giagu** for the "Machine Learning in Physics" course at Sapienza University of Rome / CERN.

Due to course policies and the educational nature of the dataset:
* The **source code** is fully available in this repository.
* The **raw data files (`.npz`)** are **not** distributed here.
* The code includes a placeholder `get_data()` function. If you have access to the specific CERN/Sapienza servers, you can restore the download links to reproduce the results.

## 🚀 Usage

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/HEP_Anomaly_Detection.git](https://github.com/yourusername/HEP_Anomaly_Detection.git)
    cd HEP_Anomaly_Detection
    ```
2.  Install dependencies:
    ```bash
    pip install torch torchvision numpy matplotlib scikit-learn
    ```
3.  Run the Jupyter Notebook:
    ```bash
    jupyter notebook DiProfio_Franco_Gabellini_37.ipynb
    ```

## 👥 Authors

Project developed by:
* **[Your Name]**
* **[Colleague Name 1]**
* **[Colleague Name 2]**

*Sapienza University of Rome - Department of Physics*

---
*Based on the academic project for the Machine Learning course, A.Y. 2024/2025.*
