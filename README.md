1. GitHub repository and README

•	GitHub repository:
https://github.com/ertantherock/AI-Based-Modeling-for-Energy-Efficient-Buildings-B205-Solution/tree/main
•	README (model documentation):
https://github.com/ertantherock/AI-Based-Modeling-for-Energy-Efficient-Buildings-B205-Solution/blob/main/README.md
________________________________________
2. Main script and final model
The final competition model is implemented in:
•	kaggle(0.81).py – the exact script used to generate the final submission
Script is self-contained and:
•	Automatically extract the data ZIPs
•	Build the time-aligned sensor panel
•	Enforce the 3-hour lead time constraint
•	Train the LightGBM model
•	Generate submission.csv in the required format
________________________________________
3. Data usage and training setup
The code is designed to use:
•	2024 data: from 2024_H1_Data.zip and 2024_H2_Data.zip
•	2025 data: from 2025_H1_Data.zip
The pipeline:
•	Extracts all three ZIP bundles into an ./extracted folder
•	Extracts each RBHU-YYYY-MM.zip inside extracted/ into its own month folder
•	Uses all available months up to and including 2025-05 as the training period
(this includes the 2024 data and January–May 2025 for B205)
The target is the chilled water return temperature sensor:
•	B205WC000.AM02 (case-insensitive matching plus a check for duplicate/leaky copies)
The prediction horizon and evaluation period follow the competition description:
•	Train on January–May 2025 (plus 2024 data)
•	Predict for June and July 2025 with a 10-minute resolution
•	Output ID in "YYYY-MM-DD_HH:MM:SS" format and TARGET_VARIABLE
________________________________________
4. Time-causality and 3-hour lead
To comply with the time-causality requirement (only using data up to t − 3h to predict the target at t), the script:
•	Resamples all sensors to a 10-minute grid
•	Builds lag and rolling-window features for the most correlated sensors
•	Then shifts all features forward by 18 steps (18 × 10 minutes = 3 hours)
This means that for each prediction time t, the model only sees information from times ≤ t − 3h.
I would be happy to walk you through this part in more detail if you would like to review the implementation.
________________________________________
5. Feature engineering and model
Feature engineering:
•	Time features: minute, hour, day, day-of-week, week, month, weekend flag
•	Coverage-based sensor filtering on the training period
•	Correlation-based selection of top sensors with respect to the target
•	For selected sensors:
o	Multiple lags (e.g. 1, 2, 3, 6, 12, 36 steps)
o	Rolling means and standard deviations over multiple window sizes
Model:
•	LightGBM regressor (LGBMRegressor)
•	Objective: regression, evaluated with L1 and L2 metrics
•	Time-based split into training and validation sets (80% / 20%)
•	Early stopping and logging for validation performance
•	Final model trained on 2024 + Jan–May 2025 data for B205
The script also saves a feature_importance.png plot, showing the top features used by the model.
________________________________________
6. How to run the code
Local run (short version):
1.	Place the following ZIP files in the same folder as kaggle(0.81).py:
o	2024_H1_Data.zip
o	2024_H2_Data.zip
o	2025_H1_Data.zip
2.	Install dependencies (numpy, pandas, lightgbm, scikit-learn, matplotlib).
3.	Run:
4.	python "kaggle(0.81).py"
5.	The script will:
o	Extract and process the data
o	Train the model
o	Create submission.csv and feature_importance.png
