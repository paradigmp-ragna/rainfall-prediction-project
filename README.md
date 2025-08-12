# rainfall-prediction-project

## 📌 Overview
This project uses historical weather data to build a predictive model for rainfall occurrence.  
The goal is to analyze meteorological patterns, preprocess the dataset, train a machine learning model, and evaluate its performance in predicting whether it will rain on a given day.

## 📂 Project Structure
- **Rainfall Prediction.ipynb** – Jupyter Notebook containing the complete workflow:
  - Data loading and inspection
  - Data cleaning and preprocessing
  - Exploratory Data Analysis (EDA)
  - Feature engineering
  - Model training and evaluation
- **dataset.csv** *(not included here)* – Source dataset containing historical weather records.

## 🛠️ Technologies Used
- **Python 3**
- **Pandas** – Data manipulation and preprocessing
- **NumPy** – Numerical operations
- **Matplotlib / Seaborn** – Visualization
- **Scikit-learn** – Machine learning models & evaluation

## 📊 Workflow Summary
1. **Load Dataset** – Import weather dataset into Pandas DataFrame.
2. **Data Preprocessing** – Handle missing values, encode categorical features, normalize numerical features.
3. **EDA** – Visualize relationships between features and rainfall occurrence.
4. **Modeling** – Train and test multiple models (e.g., Logistic Regression, Decision Tree, Random Forest).
5. **Evaluation** – Compare models using accuracy, precision, recall, F1-score, and confusion matrix.
6. **Prediction** – Predict rainfall occurrence for unseen data.

## 🔍 Key Findings
- Certain weather attributes (humidity, temperature, wind speed) have strong correlation with rainfall occurrence.
- Random Forest performed best in terms of accuracy and robustness.

## 🚀 How to Run
1. Install required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
