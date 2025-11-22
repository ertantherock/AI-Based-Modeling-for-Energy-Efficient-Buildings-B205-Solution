# AI-Based Modeling for Energy-Efficient Buildings – B205 Solution

This repository contains my solution for the Bosch / Kaggle competition  
**"AI-Based Modeling for Energy-Efficient Buildings"**.

The goal is to predict the **return temperature of chilled water**  
(sensor `B205WC000.AM02`) of building **B205** with a **3-hour lead time**,  
using the building’s sensor network as input.

The solution focuses on:

- ✅ **Time-causal modeling** with a strict 3-hour look-ahead (no leakage)  
- ✅ A **single LightGBM model** with carefully engineered lag/rolling features  
- ✅ A **click-and-run script** that auto-extracts the data, builds features, trains, and writes `submission.csv`

---

## 1. Project Structure

Example layout (as in this repo):

```text
KaggleCompetition/
├── extracted/                # Created automatically – contains extracted data
│   ├── RBHU-2024-.. /        # Monthly folders created from the ZIPs
│   └── RBHU-2025-.. /
├── 2024_H1_Data.zip          # Train + extra data part 1 (2024)
├── 2024_H2_Data.zip          # Train + extra data part 2 (2024)
├── 2025_H1_Data.zip          # Train + test (Jan–Jul 2025)
├── kaggle(0.81).py           # Main script (described in this README)
└── README.md                 # This file
