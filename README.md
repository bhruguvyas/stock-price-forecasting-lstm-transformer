# Stock Price Forecasting: LSTM vs Transformer

Comparing Recurrent Neural Networks and Transformer architectures for time series forecasting using Apple Inc. (AAPL) daily stock prices.

---

## Overview

This project implements and benchmarks two deep learning approaches for univariate one-step-ahead stock price forecasting:

- **LSTM** – a 2-layer stacked recurrent network with gating mechanisms
- **Transformer** – an encoder-only model with custom sinusoidal positional encoding and multi-head self-attention (4 heads)

Both models are trained on 10 years of AAPL daily closing prices (2015–2024) downloaded via `yfinance`, using a strict temporal train/test split to prevent data leakage.

---

## Results

| Metric | LSTM | Transformer |
|---|---|---|
| MAE (USD) | 13.58 | 23.58 |
| RMSE (USD) | 17.26 | 29.27 |
| MAPE (%) | 6.03 | 10.54 |
| R² | 0.557 | −0.274 |
| Loss Reduction | 97.59% | 95.04% |
| Training Time | 30.1s | 38.7s |
| Parameters | 116,801 | 102,209 |

> **Key finding**: LSTM outperformed the Transformer on all metrics for this dataset and sequence length (30 days). The Transformer's negative R² indicates it failed to generalise to the 2024 test period, where prices exceeded the MinMaxScaler's training range — a known extrapolation limitation.

---

## Architecture

### LSTM
```
Input (batch, 30, 1)
  → LSTM Layer 1 : 128 hidden units  (returns all time steps)
  → Dropout (0.2)
  → LSTM Layer 2 : 64 hidden units   (returns last time step)
  → Dropout (0.2)
  → Linear       : 64 → 1
```

### Transformer
```
Input (batch, 30, 1)
  → Linear Projection    : 1 → 64 (d_model)
  → Positional Encoding  : sinusoidal, custom implementation
  → TransformerEncoder   : 2 layers, nhead=4, dim_ff=256
  → Last time step       : (batch, 64)
  → Output Head          : Linear 64→32 → ReLU → Linear 32→1
```

### Positional Encoding (Vaswani et al., 2017)

```
PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
```

---

## Project Structure

```
stock-price-forecasting-lstm-transformer/
│
├── stock-price-forecasting-lstm-transformer.ipynb   # Main notebook (fully executed)
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Setup and Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/stock-price-forecasting-lstm-transformer.git
cd stock-price-forecasting-lstm-transformer
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebook
```bash
jupyter notebook stock-price-forecasting-lstm-transformer.ipynb
```

Then: **Kernel → Restart & Run All**

> The notebook downloads AAPL data automatically via `yfinance` — no manual dataset download needed.

---

## Dataset

- **Source**: Yahoo Finance via `yfinance` library
- **Ticker**: AAPL (Apple Inc.)
- **Period**: 2015-01-01 to 2024-12-31
- **Frequency**: Daily closing prices
- **Total points**: 2,515 trading days
- **Train / Test split**: 90 / 10 (chronological, no shuffling)
- **Sequence length**: 30 days lookback → predict 1 day ahead

---

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 80 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimiser | Adam (weight_decay=1e-5) |
| Loss function | MSELoss |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Gradient clipping | max_norm=1.0 |
| Normalisation | MinMaxScaler (fit on train only) |
| Framework | PyTorch 2.10.0 |
| Device | CUDA (Tesla T4) |

---

## Key Implementation Details

- **No data leakage**: scaler fitted exclusively on training data, then applied to test set
- **No shuffling**: DataLoader uses `shuffle=False` — temporal order preserved throughout
- **No pre-trained models**: entire architecture implemented from scratch in PyTorch
- **Inverse transformation**: all metrics reported on original USD price scale
- **Gradient clipping**: applied at `max_norm=1.0` to prevent exploding gradients in LSTM

---

## Evaluation Metrics

All four metrics computed on the original (inverse-transformed) price scale:

| Metric | Formula | What it tells you |
|---|---|---|
| MAE | mean(&#124;y − ŷ&#124;) | Average dollar error |
| RMSE | sqrt(mean((y − ŷ)²)) | Error with heavier penalty on large mistakes |
| MAPE | mean(&#124;y − ŷ&#124; / y) × 100 | Scale-independent % error |
| R² | 1 − SS_res/SS_tot | How much better than always predicting the mean |

**Primary metric: RMSE** — chosen because large prediction errors carry disproportionate financial risk in stock price forecasting.

---

## Visualisations

The notebook includes:
- AAPL price history with EDA (rolling mean, return distribution)
- Train/test split visualisation
- Sinusoidal positional encoding heatmap
- Training loss curves (both models) with reduction annotations
- Actual vs Predicted price plots (with error ribbons)
- Scatter plots (predicted vs actual)
- Residual plots (over time + distribution)
- Side-by-side metric comparison bar chart

---

## Environment

```
Python       : 3.12.13
PyTorch      : 2.10.0+cu128
NumPy        : 2.0.2
Pandas       : 2.2.2
scikit-learn : 1.6.1
yfinance     : 0.2.66
CUDA         : Tesla T4
OS           : Linux 6.6.113+
```

---

## References

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.
- PyTorch Documentation: [pytorch.org/docs](https://pytorch.org/docs/stable/index.html)

---

## 👤 Author

**Bhrugu Vyas**  
Senior Data Analyst - JP Morgan Chase    
MTech AI/ML — BITS Pilani

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/bhruguvyas/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/bhruguvyas)
