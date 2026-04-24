# 🎵 Variational Autoencoders for Hybrid Language Music Clustering

## Dataset LINK https://drive.google.com/drive/folders/1-f8qxqddDUtRO_pu3nwaCruBnEDI9ENg?usp=sharing

## 📌 Overview
This project explores **unsupervised music clustering** using **Variational Autoencoders (VAE)** and **Beta-VAE** on hybrid-language music datasets. The goal is to understand how well deep latent representations can cluster music when combining **audio features and lyrics**.

The project evaluates multiple approaches across increasing levels of complexity and compares deep learning methods with traditional baselines.

---

## 🚀 Key Contributions
- 🔹 Multimodal clustering using **audio + lyric feature fusion**
- 🔹 Implementation of **VAE, Beta-VAE, Conv-VAE, Autoencoder, and PCA**
- 🔹 Evaluation on both **hybrid Bengali-English dataset** and **GTZAN benchmark**
- 🔹 Comparison with strong baselines like **MFCC + K-Means**
- 🔹 Analysis of **latent space structure using UMAP**
- 🔹 Tuning of **β (beta) parameter** for improved clustering performance

---

## 📊 Tasks & Methodology

### 🔹 Low Complexity
- Input: Tabular audio features  
- Models:
  - VAE + K-Means
  - PCA + K-Means  
- **Best Result:** VAE + K-Means

---

### 🔹 Medium Complexity

#### Hybrid (Audio + Lyrics)
- Feature fusion (Audio + TF-IDF + SVD)
- Models:
  - VAE + K-Means
  - Agglomerative Clustering
  - DBSCAN  
- **Best Result:** K-Means on fused latent features

#### GTZAN (Audio Only)
- Models:
  - Conv-VAE on Mel-Spectrograms
  - MFCC + K-Means  
- **Best Result:** MFCC + K-Means

---

### 🔹 High Complexity
- Input: Cleaned fused audio + lyrics dataset  
- Models:
  - Beta-VAE + K-Means (tuned)
  - Autoencoder + K-Means
  - PCA + K-Means  

- **Final Model:**  
  ✅ Tuned **Beta-VAE (β = 1.0) + K-Means**

---

## 📈 Results Summary

| Task   | Best Method | Key Metric |
|--------|------------|-----------|
| Low    | VAE + K-Means | Silhouette: 0.3515 |
| Medium (Hybrid) | K-Means (Fused Latent) | Silhouette: 0.3023 |
| Medium (GTZAN) | MFCC + K-Means | NMI: 0.3050 |
| High   | Beta-VAE (β=1.0) + K-Means | Purity: 0.4446 |

---

## 🧠 Key Insights
- VAE models outperform PCA in capturing **nonlinear structure**
- Multimodal fusion improves clustering in hybrid datasets
- MFCC remains a strong baseline for **genre-based audio tasks**
- Lower β improves **reconstruction–clustering balance**
- Latent space visualizations reveal **language-level separation**

---

## 🛠️ Tech Stack
- Python
- PyTorch
- Scikit-learn
- Librosa
- Pandas / NumPy
- Matplotlib / Seaborn

---


---

## 📸 Visualizations
- UMAP projections of latent spaces
- Cluster distributions
- Reconstruction comparisons (Beta-VAE)
- GTZAN genre overlap analysis

---

## 📚 References
Key concepts used in this project:
- Variational Autoencoders (VAE)
- Beta-VAE
- MFCC features
- Deep clustering methods

---

## 🎯 Future Work
- Use **self-supervised audio models (wav2vec, HuBERT)**
- Apply **transformer-based audio encoders**
- Explore **deep clustering methods (DEC, contrastive learning)**
- Scale to **larger multilingual datasets**

---

## 👨‍💻 Author
**Arko Mazhar**  
Student ID: 23141091  
BRAC University  
📧 arko.mazhar@g.bracu.ac.bd

---
