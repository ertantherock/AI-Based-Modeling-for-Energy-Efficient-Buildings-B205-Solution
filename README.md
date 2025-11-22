# AI-Based Modeling for Energy-Efficient Buildings – B205 Solution

This repository contains my solution for the Bosch / Kaggle competition  
**“AI-Based Modeling for Energy-Efficient Buildings”**.

The competition goal is to **predict the building’s cooling load**, approximated by the  
**return temperature of chilled water** (`B205WC000.AM02`), for **building B205** with a strict  
**3-hour lead time**, using hundreds of sensor time series from the Bosch Budapest Campus building.

This repository provides:

- A **single, auditable Python script** – `kaggle(0.81).py` – which corresponds to my final
  competition model (top result on the private leaderboard / winning model).
- An additional script `kaggle_v2.py` that is a slightly refactored version of the same pipeline.
- A fully reproducible workflow that:
  - Extracts the data bundles (2024 + 2025)
  - Builds a clean 10-minute sensor panel
  - Enforces the **3-hour time-causality constraint**
  - Trains a **LightGBM** model
  - Generates a competition-ready `submission.csv`

---

## 1. Competition Context

### 1.1 Target Variable

- Sensor: **`B205WC000.AM02`**
- Meaning: *RETURN TEMPERATURE CHILLED WATER*  
  → used as a proxy for the **cooling load** of building B205.

### 1.2 Data & Time Ranges

- **Training period (competition definition):**
  - January–May 2025 (`RBHU-2025-01` … `RBHU-2025-05`)
- **Additional training data (allowed by competition):**
  - All available months from **2024** (from `2024_H1_Data.zip` and `2024_H2_Data.zip`)
- **Evaluation / Test period:**
  - June–July 2025
    - June → public leaderboard
    - July → private leaderboard
- **Sampling period:** 10 minutes.

### 1.3 Task Type

- Not a pure target-only time-series forecast.
- Instead: learn the **relationship between the other variables and the target**,  
  under a strict **time-causal + 3-hour lead time** requirement.

### 1.4 Causality / Lead Time

- To predict the target at time **t**, the model may only use information up to **t − 3 hours**.
- In this solution, this is enforced by **shifting all features 18 steps forward**  
  (`18 × 10 minutes = 3 hours`).

### 1.5 Evaluation Metric

- **Mean Squared Error (MSE)** between predicted and true target values over the period  
  June–July 2025.

---

## 2. Data Usage in This Code

The final model (`kaggle(0.81).py`) is designed to use **both 2024 and 2025 data**:

- It looks for three bundles in the script directory:
  - `2024_H1_Data.zip`
  - `2024_H2_Data.zip`
  - `2025_H1_Data.zip`
- If the 2024 ZIPs are present, **all 2024 months** (RBHU-2024-..) are included in training.
- Training months are defined as all `RBHU-YYYY-MM` folders where `(YYYY, MM) <= (2025, 5)`,  
  i.e.:
  - **All 2024 months**, plus  
  - January–May 2025  
- The evaluation / prediction period is June–July 2025 only.

If the 2024 zip files are missing, the pipeline still works and trains only on **Jan–May 2025**.

---

## 3. Repository Structure

Example layout (matching the project setup):

```text
KaggleCompetition/
├── extracted/                # Auto-created – contains extracted monthly data
│   ├── RBHU-2024-.. /        # Monthly folders created from the data zips
│   └── RBHU-2025-.. /
├── 2024_H1_Data.zip          # Original Bosch/Kaggle data bundle, part 1 (2024)
├── 2024_H2_Data.zip          # Original Bosch/Kaggle data bundle, part 2 (2024)
├── 2025_H1_Data.zip          # Original Bosch/Kaggle data bundle (2025)
├── feature_importance.png    # Saved after training (top 30 features)
├── kaggle(0.81).py           # Final competition model (main script)
├── kaggle_v2.py              # Refactored version of the same pipeline
├── submission.csv            # Generated Kaggle submission
└── README.md                 # This file
