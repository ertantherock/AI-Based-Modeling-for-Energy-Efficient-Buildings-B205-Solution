## 1. GitHub Repository and Documentation

- **GitHub repository**  
  👉 [AI-Based Modeling for Energy-Efficient Buildings – B205 Solution](https://github.com/ertantherock/AI-Based-Modeling-for-Energy-Efficient-Buildings-B205-Solution/tree/main)

- **README (model documentation)**  
  👉 [README.md](https://github.com/ertantherock/AI-Based-Modeling-for-Energy-Efficient-Buildings-B205-Solution/blob/main/README.md)

---

## 2. Main Script and Final Model

The final competition model is implemented in:

- **`kaggle(0.81).py`** – the exact script used to generate the final submission.

This script is **fully self-contained** and:

- Automatically **extracts the data ZIPs**.
- Builds a **time-aligned sensor panel** at 10-minute resolution.
- Enforces the **3-hour lead time** constraint.
- Trains a **LightGBM** regression model.
- Generates **`submission.csv`** in the required competition format.

---

## 3. Data Usage and Training Setup

The code is designed to use the following data:

- **2024 data**
  - `2024_H1_Data.zip`
  - `2024_H2_Data.zip`
- **2025 data**
  - `2025_H1_Data.zip`

### Data pipeline

- Extracts all three ZIP bundles into an `./extracted` folder.
- Inside `extracted/`, extracts each `RBHU-YYYY-MM.zip` into its own month folder.
- Uses **all available months up to and including 2025-05** as the **training period**  
  → this includes **all 2024 data** and **January–May 2025** for building **B205**.

### Target variable

- **Target sensor:** `B205WC000.AM02`  
- The script:
  - Matches this sensor **case-insensitively**.
  - Checks for and removes **duplicate / leaky copies** of the target.

### Prediction horizon & evaluation period

The setup follows the competition description:

- Train on **2024 + January–May 2025**.
- Predict for **June and July 2025** with **10-minute resolution**.
- Output:
  - `ID` in `"YYYY-MM-DD_HH:MM:SS"` format.
  - `TARGET_VARIABLE` with the model’s prediction.

---

## 4. Time-Causality and 3-Hour Lead

To comply with the time-causality requirement  
(*only using data up to t − 3h to predict the target at time t*), the script:

- Resamples all sensors to a **10-minute grid**.
- Builds **lag** and **rolling-window** features for the most correlated sensors.
- Then **shifts all features forward by 18 steps**  
  (`18 × 10 minutes = 3 hours`).

As a result:

> For each prediction time **t**, the model only sees information from times **≤ t − 3h**.

This strictly enforces the 3-hour lead-time requirement from the competition.

---

## 5. Feature Engineering and Model

### Feature engineering

The script performs several feature engineering steps:

- **Time-based features**
  - `minute`, `hour`, `day`, `day_of_week`, `week`, `month`, `is_weekend`.

- **Sensor selection & filtering**
  - Coverage-based filtering on the **training period**.
  - Correlation-based selection of **top sensors** with respect to the target.

- **For selected sensors**, it creates:
  - Multiple **lags** (e.g. 1, 2, 3, 6, 12, 36 steps → 10-minute steps).
  - **Rolling means** and **rolling standard deviations** over multiple window sizes  
    (e.g. 3, 6, 12, 36 steps).

### Model

- **Model type:** LightGBM regressor (`LGBMRegressor`).
- **Objective:** regression.
- **Metrics monitored:** L1 and L2 loss (MAE and MSE/RMSE).
- **Train/validation split:**
  - Time-based split: **80%** for training, **20%** for validation (no shuffling).
- **Training tricks:**
  - Early stopping on the validation set.
  - Logging of validation performance during training.

The **final model** is trained on:

- **All 2024 data** for B205, plus  
- **January–May 2025** data for B205.

The script also saves:

- **`feature_importance.png`** – a bar plot of the **top features** used by LightGBM.

---

## 6. How to Run the Code

### Local run (short version)

1. **Place the ZIP files** in the same folder as `kaggle(0.81).py`:

   - `2024_H1_Data.zip`  
   - `2024_H2_Data.zip`  
   - `2025_H1_Data.zip`

2. **Install dependencies** (for example via pip):

   ```bash
   pip install numpy pandas lightgbm scikit-learn matplotlib
### What the Script Does

When you run:

    python "kaggle(0.81).py"

the script will:

- **Extract and process the data**
  - Unzip the 2024 and 2025 data bundles
  - Build a time-aligned 10-minute sensor panel for building B205

- **Train the LightGBM model**
  - Generate features (lags, rolling windows, time features)
  - Fit a LightGBM regressor with a time-based train/validation split

- **Create the following outputs**
  - `submission.csv` – ready to submit to the competition  
  - `feature_importance.png` – visualization of the top model features
