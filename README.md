# Anomaly Detection in High-Energy Physics using Deep Autoencoders

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 🔬 Project Overview

This project applies **Deep Learning** techniques to a fundamental problem in High-Energy Physics (HEP): detecting anomalous hadronic jets produced at the **Large Hadron Collider (LHC)**.

We implemented a **Convolutional Autoencoder (CNN-AE)** to perform unsupervised anomaly detection, training the model to reconstruct "normal" jets and identifying anomalies based on reconstruction error and latent space analysis.

## 🎯 Key Objectives

* **Unsupervised Learning:** Train a model exclusively on Standard Model background data (QCD jets).
* **Architecture:** Design a Convolutional Autoencoder capable of compressing 2D calorimeter images into a dense latent representation.
* **Anomaly Scoring:** Compare different metrics for outlier detection:
    * **MSE (Reconstruction Loss):** $L = ||x - \hat{x}||^2$
    * **Mahalanobis Distance:** Calculated in the latent space.
* **Latent Space Analysis:** Use of **PCA**, **UMAP** and **Gaussian Mixture Models (GMM)** to interpret how the network organizes physical features.

## 📊 Results

### 1. Reconstruction Loss Analysis
A reconstruction loss threshold is imposed such that the FPR is of 10% for the "Normal" dataset. The results obtained by using the latter threshold on the mixed datasets are compatible with the expected anomaly rates, though a lack of two visibly separated peaks for the normal and anomalous events is noted. More statisfactory results are obtained with the classification based on the Mahalanobis distance.

### 2. Latent Space Visualization
By projecting the latent space onto 2D using **PCA**, we observed distinct clustering behaviors between the background and the anomalous signals, validating the model's feature extraction capabilities. Analogous results are obtained using **UMAP**

### 3. Gaussian Mixture Models Analysis
By setting the number of gaussian clusters to 2, being the "normal" and "anomalous" cluster, the anomaly rates result once more compatible with what expected. A purity analysis between the GMM's labels and the ones obtained with the reconstruction loss and the Mahalanobis distance result much higher for the latter.


### 3. Performance
* **False Positive Rate (FPR):** Maintained $\le 10\%$ on background data.
* **Signal Discrimination:** We confirmed the physical hypothesis that higher mass particles ($f_{high}$) result in higher reconstruction errors compared to lower mass anomalies ($f_{low}$), i.e., $f_{high} > f_{low}$.

## 📂 Data Availability

The dataset used in this project consists of simulated LHC events provided by **Prof. Stefano Giagu** for the "Machine Learning in Physics" course at Sapienza University of Rome / CERN.

Due to course policies and the educational nature of the dataset:
* The **source code** is fully available in this repository.
* The **raw data files (`.npz`)** are **not** distributed here.
* The code includes a placeholder `get_data()` function. If you have access to the specific CERN/Sapienza servers, you can restore the download links to reproduce the results.

## 👥 Authors

Project developed by:
* **Lorenzo Gabellini**
* **Caterina Di Profio**
* **Giuseppe Franco**

*Sapienza University of Rome - Department of Physics*

---
*Based on the academic project for the Machine Learning course, A.Y. 2024/2025.*
