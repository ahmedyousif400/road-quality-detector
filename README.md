# Road Quality Detector - Random Forest Classifier

## Overview
This project implements a **Random Forest Classifier** to detect and classify road quality conditions. The model analyzes road surface characteristics and predicts quality levels (e.g., Good, Fair, Poor) to help with infrastructure maintenance and planning.

## Project Objective
- Classify road conditions based on various surface features
- Provide automated road quality assessment for maintenance prioritization
- Use ensemble learning for robust and accurate predictions

## Dataset
The model is trained on road surface data with features such as:
- **Pavement Condition Index (PCI)**
- **Roughness measurements**
- **Pothole detection**
- **Crack density**
- **Surface distress indicators**
- **Age of road surface**
- **Traffic intensity**

**Dataset Shape:** [Specify your dataset dimensions]  
**Target Variable:** Road Quality (Classes: Good/Fair/Poor or similar)

## Requirements
```python
# Core Libraries
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0

# Optional (for enhanced visualization)
plotly>=5.0.0
```

**Installation:**
```bash
pip install -r requirements.txt
```

## Project Structure
```
road-quality-detector/
├── data/
│   ├── raw_data.csv              # Original dataset
│   └── processed_data.csv        # Cleaned & processed data
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA
│   ├── 02_data_preprocessing.ipynb     # Data cleaning & feature engineering
│   ├── 03_model_training.ipynb         # Random Forest model training
│   └── 04_model_evaluation.ipynb       # Model performance & validation
├── models/
│   └── random_forest_model.pkl   # Trained model
├── results/
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── README.md
└── requirements.txt
```

## Quick Start

### 1. **Data Exploration**
Open `notebooks/01_data_exploration.ipynb` to:
- Load and inspect the dataset
- Check for missing values
- Visualize feature distributions
- Analyze class balance

### 2. **Data Preprocessing**
Run `notebooks/02_data_preprocessing.ipynb` to:
- Handle missing values
- Encode categorical features
- Scale numerical features (if needed)
- Split data into train/test sets

### 3. **Model Training**
Execute `notebooks/03_model_training.ipynb` to:
- Initialize Random Forest Classifier
- Train the model on training data
- Tune hyperparameters
- Save the trained model

### 4. **Model Evaluation**
Run `notebooks/04_model_evaluation.ipynb` to:
- Evaluate model performance on test data
- Generate confusion matrix
- Calculate precision, recall, F1-score
- Visualize feature importance

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | [Your Accuracy]% |
| Precision | [Your Precision]% |
| Recall | [Your Recall]% |
| F1-Score | [Your F1-Score] |

### Feature Importance (Top 5)
```
1. [Feature Name] - X%
2. [Feature Name] - X%
3. [Feature Name] - X%
4. [Feature Name] - X%
5. [Feature Name] - X%
```

## Random Forest Classifier Details

**Hyperparameters Used:**
```python
RandomForestClassifier(
    n_estimators=100,           # Number of trees
    max_depth=10,               # Maximum tree depth
    min_samples_split=20,        # Minimum samples for split
    min_samples_leaf=10,         # Minimum samples per leaf
    random_state=42,            # For reproducibility
    n_jobs=-1                   # Parallel processing
)
```

**Why Random Forest?**
-  Handles both categorical and numerical features
-  Robust to outliers and missing values
-  Provides feature importance rankings
-  Reduces overfitting through ensemble approach
-  Fast inference time for real-time predictions

## Key Insights

- **Feature Importance:** [Describe which features matter most]
- **Model Strengths:** [List model advantages for your use case]
- **Limitations:** [Mention any constraints or areas for improvement]

## Making Predictions

```python
import pickle
import pandas as pd

# Load trained model
with open('models/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare new data
new_data = pd.DataFrame({
    'feature1': [value],
    'feature2': [value],
    # ... other features
})

# Make prediction
prediction = model.predict(new_data)
prediction_proba = model.predict_proba(new_data)

print(f"Predicted Road Quality: {prediction[0]}")
print(f"Confidence: {max(prediction_proba[0]) * 100:.2f}%")
```

## Jupyter Notebook Best Practices Used

-  Clear cell documentation with markdown
-  Step-by-step data processing pipeline
-  Visualizations at each stage
-  Model training with progress tracking
-  Comprehensive evaluation metrics
-  Reproducible code with seed values

## Model Validation

- **Cross-Validation:** K-Fold cross-validation (k=5) used for robust evaluation
- **Train-Test Split:** 80% training, 20% testing

## Future Improvements

- [ ] Integrate additional road condition sensors
- [ ] Implement deep learning approaches for comparison
- [ ] Deploy model as a web service
- [ ] Add real-time prediction capability
- [ ] Collect more diverse road condition data

## Notes for Submission

- All notebooks are self-contained and can be run independently
- Ensure `data/` folder contains your dataset before running
- Model training may take a few minutes depending on dataset size
- Results are saved to `results/` folder for reference

## Author
**Ahmed Yousif Saadeldeen**  
College Assignment - Road Quality Detection  
Date: 6/20/2026

## License
This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

You may copy, distribute, and modify the software as long as you track changes/dates in source files. Any modifications to or software including (via compiler) GPL-licensed code must also be made available under the GPL along with build and install instructions. See the LICENSE file for more details.

---

**Happy analyzing! 🚗🛣️**
