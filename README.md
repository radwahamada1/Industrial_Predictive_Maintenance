# Industrial Predictive Maintenance — RUL Predictor

An end-to-end **Deep Learning system** for predicting the **Remaining Useful Life (RUL)** of industrial turbofan engines using time-series sensor data from the **NASA C-MAPSS dataset**.

The project uses an **LSTM neural network** to learn engine degradation patterns and estimate the number of operational cycles remaining before failure.

---

## Project Overview

Predictive maintenance helps reduce unexpected equipment failures and optimize maintenance schedules.

**Input:** Historical sensor measurements
**Output:** Predicted Remaining Useful Life (RUL)

---

## Tech Stack


<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio"/>
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face"/>
</p>
**Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-Learn · TensorFlow/Keras · Gradio · Hugging Face**
**Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-Learn · TensorFlow/Keras · Gradio · Hugging Face**

---

## Methodology

### 1. Data Preparation

* Loaded and analyzed the NASA C-MAPSS dataset.
* Inspected data structure, dimensions, and data types.
* Checked for missing values.
* Removed constant and uninformative sensors.

### 2. Exploratory Data Analysis

* Analyzed sensor distributions and degradation patterns.
* Visualized sensor behavior across operating cycles.
* Used correlation analysis to understand relationships between features.

### 3. Feature Engineering

* Generated the **RUL target** for each engine cycle.
* Applied RUL capping at **125 cycles**.
* Scaled sensor features using `MinMaxScaler`.
* Created sliding-window sequences for time-series modeling.

### 4. LSTM Model

* Built a deep LSTM architecture with Dropout layers.
* Used the **Adam optimizer** and **MSE loss**.
* Applied **Early Stopping** to reduce overfitting and restore the best model weights.

### 5. Evaluation

The model was evaluated using:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**
* Actual vs. Predicted RUL visualization
* Residual analysis

### 6. Deployment

The trained model was integrated into an interactive **Gradio** application and deployed using **Hugging Face Hub and Spaces**.

---

## Project Pipeline

```text
NASA C-MAPSS Dataset
        ↓
Data Cleaning & EDA
        ↓
Feature Engineering
        ↓
RUL Generation
        ↓
Scaling & Sliding Windows
        ↓
LSTM Model
        ↓
Model Evaluation
        ↓
Gradio Application
        ↓
Hugging Face Deployment
```

---

## Live Demo

Try the deployed application:

**[Launch the Live Application](https://radwahamada1-industrial-app.hf.space/?__theme=system&deep_link=-QYP-utyM-I)**

The application provides an interactive **Gradio interface** for real-time RUL prediction.

---

## Project Status

**Ongoing Development**

This project is still under active development. The current version focuses on building and deploying the core RUL prediction pipeline, with further improvements and experiments being added continuously.
