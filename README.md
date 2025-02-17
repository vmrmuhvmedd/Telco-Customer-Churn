# Telco Customer Churn Prediction

## Overview
This project aims to predict customer churn using machine learning techniques. The dataset used is the "Telco Customer Churn" dataset, which contains customer information and whether they have churned. The project uses Logistic Regression with Label Encoding for categorical variables and includes model evaluation metrics, hyperparameter tuning, and feature importance analysis.

## Features
- **Data Preprocessing**: Handles missing values and encodes categorical features using Label Encoding.
- **Exploratory Data Analysis (EDA)**: Provides insights into customer churn through visualizations.
- **Feature Scaling**: Uses StandardScaler to normalize numerical data.
- **Model Training**: Implements Logistic Regression with hyperparameter tuning using GridSearchCV.
- **Model Evaluation**: Computes accuracy, precision, recall, F1-score, ROC-AUC score, and plots confusion matrix & ROC curve.
- **Feature Importance Analysis**: Identifies key factors influencing customer churn.

## Dataset
The dataset used is `WA_Fn-UseC_-Telco-Customer-Churn.csv`. It includes customer demographic details, account information, and churn status.

## Installation
Clone the repository:
```bash
git clone https://github.com/your-username/telco-churn-prediction.git
cd telco-churn-prediction
```
Install required dependencies:
```bash
pip install -r requirements.txt
```
Run the script:
```bash
python churn_prediction.py
```

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage
- Modify `churn_prediction.py` to experiment with different models or preprocessing techniques.
- Run the script to train the model and evaluate its performance.
- View the generated plots for insights into customer churn.

## Results
- Achieved high accuracy using Logistic Regression with hyperparameter tuning.
- Identified key features contributing to customer churn.
- Visualized churn patterns through EDA.

## Contributions
Feel free to fork this repository, make improvements, and submit pull requests.

## License
This project is licensed under the MIT License.
