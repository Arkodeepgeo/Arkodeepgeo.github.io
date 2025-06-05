---
layout: default
title: "Machine Learning for Zn-Pb Deposit Classification"
---

# 🧪 Classification of Zn-Pb Deposits Using Sphalerite Geochemistry

Welcome to the official project page of **Arkodeep Sengupta**. This work presents the application of machine learning models to classify **Zn-Pb deposit types** based on trace element chemistry of sphalerite minerals. The results demonstrate promising accuracy and are part of a broader geochemical exploration study.

> 📍 _"Geochemistry meets Machine Learning in Mineral Exploration."_

---

## 🧬 Project Overview

The goal of this project is to distinguish between different Zn-Pb deposit types using geochemical signatures. Sphalerite, a common ore mineral, hosts a range of trace elements that reflect the conditions and sources of ore formation. We use these trace element patterns to train machine learning models for deposit classification.

**Deposit types studied:**
- SEDEX
- MVT
- VMS
- Epithermal
- Skarn

---

## 💻 Tools & Methods

**Data Source:**
- 5018 sphalerite samples from 115 Zn-Pb deposits.

**Preprocessing:**
- Outlier removal
- Centered log-ratio (CLR) transformation
- Standard scaling

**Models Used:**
- Decision Tree (DT)
- Naive Bayes (NB)
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Multi-layer Perceptron (MLP)
- Random Forest (RF)
- XGBoost
- Bagging & Boosting (CART-based)
- Stacking Classifier (RF + XGB, meta = Multinomial Logistic Regression)

**Hyperparameter Tuning:**
- Bayesian Optimization (for SVM, RF, XGBoost)

---

## 📊 Visualizations

### 🔁 Radar Chart of Model Performance

![Radar Plot](Radar%20Plot.jpg)

This radar plot compares the classification performance of various models across key metrics: accuracy, F1 score, kappa, and MCC.

### 🧾 Confusion Matrices

![Confusion Matrices](Confusion%20Matrices.jpg)

A set of confusion matrices for each model, showing how well each deposit type is predicted.

---

## 🔍 Insights

- **SVM and XGBoost** deliver the highest overall classification accuracy (up to ~97%).
- **Epithermal deposits** remain the most misclassified group, often confused with MVT.
- Ensemble techniques (stacking, boosting) outperform individual learners.
- The geochemical variability of trace elements like Fe, Mn, In, and Ga effectively contributes to distinguishing deposit types.

---

## 📁 Dataset & Code

- **Dataset:** Not publicly available due to publication constraints.
- **Code Repository:** [GitHub Repo](https://github.com/Arkodeepgeo/ZnPb-ML)

---

## 🌐 Web App

Check out the interactive R Shiny app:  
▶️ [Zn-Pb Deposit Classifier](https://arkodeepgeo.shinyapps.io/znpb-classifier)

---

## 📚 References

- Guilbert, J.M., & Park, Jr., C.F. (2007). *The Geology of Ore Deposits*.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
- Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System.
- Kuhn, M. (2008). Building Predictive Models in R Using the caret Package.

---

## 👨‍🔬 Author

**Arkodeep Sengupta**  
_MTech (Exploration Geology), Machine Learning Enthusiast_  
📧 arkodeep.geo@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/arkodeepgeo) | [GitHub](https://github.com/Arkodeepgeo)

---

