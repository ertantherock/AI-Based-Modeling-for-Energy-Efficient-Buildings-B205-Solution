# AI-Based Modeling for Energy-Efficient Buildings – B205 Solution

This repository contains my solution for the Bosch / Kaggle competition  
**“AI-Based Modeling for Energy-Efficient Buildings”**.

The competition goal is to **predict the building’s cooling load**, approximated by the  
**return temperature of chilled water** (`B205WC000.AM02`), for **building B205** with a strict  
**3-hour lead time**, using hundreds of building sensor time series.

This repository provides:

- A **single, auditable Python script** – `kaggle(0.81).py` – which corresponds to my final
  competition model (top-4 on the private leaderboard).
- An additional script `kaggle_v2.py` that is a slightly cleaned version of the pipeline.
- A fully reproducible workflow that:
  - Extracts the data bundles
  - Builds a clean 10-minute sensor panel
  - Enforces the **3-hour time-causality constraint**
  - Trains a **LightGBM** model
  - Generates a competition-ready `submission.csv`

---

## 1. Competition Context

### 1.1 Target Variable

- Sensor: **`B205WC000.AM02`**
- Meaning: *RETURN TEMPERATURE CHILLED WATER*, used as a proxy for the building’s **cooling load**.

### 1.2 Data & Time Ranges

- **Training period:** January–May 2025  
  (`RBHU-2025-01` … `RBHU-2025-05`).
- **Evaluation / Test period:** June–July 2025  
  - June → public leaderboard  
  - July → private leaderboard
- **Sampling period:** 10 minutes.

### 1.3 Task Type

- Not a pure time-series forecast of the target alone.
- Instead: learn the **relationship between all other variables and the target**,  
  under a strict **time-causal + 3-hour lead time** requirement.

### 1.4 Causality / Lead Time

- To predict the target at time **t**, the model may only use information up to **t − 3 hours**.
- In this solution, this is enforced by **shifting all features 18 steps forward**  
  (`18 × 10 minutes`).

### 1.5 Evaluation Metric

- **Mean Squared Error (MSE)** between predicted and true values over June–July 2025.

### 1.6 Dataset Description (short)

- Time-series sensor data for the Bosch Budapest Campus buildings (B205, B201, etc.), including:
  - Temperatures, humidity, air quality, flow, energy consumption
  - Valve and damper positions, pump/fan states
  - Setpoints, controller outputs, switches, alarms, and more
- Original sampling is **non-uniform**; this solution resamples everything to a **10-minute grid**.
- Metadata:
  - `metadata.parquet` / `metadata.xlsx` with:
    - `object_id`, `class_id`, `description`, `unit`, `bde_channel_typ`, device and channel identifiers, etc.
  - Used mainly for interpretation; the core pipeline does not require it.

---

## 2. Repository Structure

Example layout (matching the VS Code screenshot):

```text
KaggleCompetition/
├── extracted/                # Auto-created – contains extracted monthly data
│   ├── RBHU-2024-.. /        # Monthly folders created from the data zips
│   └── RBHU-2025-.. /
├── 2024_H1_Data.zip          # Original Bosch/Kaggle data bundle, part 1 (2024)
├── 2024_H2_Data.zip          # Original Bosch/Kaggle data bundle, part 2 (2024)
├── 2025_H1_Data.zip          # Original Bosch/Kaggle data bundle (2025)
├── feature_importance.png    # Saved after training (top 30 features)
├── kaggle_v2.py              # Refactored version of the pipeline
├── kaggle(0.81).py           # Final competition model (main script)
├── submission.csv            # Generated Kaggle submission
└── README.md                 # This file
