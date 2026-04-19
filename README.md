# 📊 Hybrid ARIMA–LSTM–GARCH Forex Forecasting Model

## 📌 Overview
This project implements a hybrid time-series forecasting system for Forex price prediction using:
- ARIMA → captures linear trends  
- LSTM → captures nonlinear patterns  
- GARCH → models volatility and uncertainty  

---

## ⚙️ Features
- ADF stationarity testing
- ARIMA auto selection
- LSTM deep learning model
- Feature engineering
- Weighted hybrid model
- GARCH volatility modeling
- Confidence intervals
- Full output saving

---

## 📁 Project Structure
project/
│
├── Forex_OHLC_50k_Dataset.xlsx
├── hybrid_outputs/
└── main.py

---

## 📊 Dataset Requirements
Required columns:
- Open
- High
- Low
- Close
- Volume (optional)

---

## 🧠 Model Architecture

### ARIMA
- Finds optimal (p, d, q)
- Models linear components

### LSTM
- Sequence window: 60
- Learns nonlinear patterns

### Hybrid
Hybrid = w1 * ARIMA + w2 * LSTM

### GARCH
Models volatility and builds confidence intervals

---

## 🚀 How to Run

1. Install dependencies:
pip install numpy pandas matplotlib scikit-learn statsmodels arch tensorflow pmdarima

2. Update dataset path in code

3. Run:
python main.py

---

## 📈 Outputs
- Prediction CSV
- Performance metrics
- Training history
- Plots and figures

---

## 📊 Evaluation
Metrics:
- MAE
- RMSE

---

## 🎯 Advantages
- Combines statistical + deep learning
- Captures trend + nonlinear patterns
- Includes risk estimation

---

## ⚠️ Limitations
- Sensitive to hyperparameters
- Requires large dataset
- Training time can be high

---

## 📌 Summary
A hybrid model for accurate Forex prediction with volatility estimation.
