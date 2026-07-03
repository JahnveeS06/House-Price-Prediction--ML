# 🏠 House Price Prediction using Machine Learning

A machine learning project that predicts house prices using the Ames Housing dataset. This project covers the complete ML workflow including data exploration, preprocessing, feature engineering, model training, evaluation, and comparison of multiple regression algorithms.

---

## 📌 Project Overview

The goal of this project is to build a regression model capable of predicting house sale prices based on various property features.

The project includes:

- Exploratory Data Analysis (EDA)
- Missing value handling
- Outlier removal
- Log transformation of the target variable
- Categorical feature encoding
- Training multiple regression models
- Model evaluation and comparison

---

## 📂 Dataset

- **Dataset:** Ames Housing Dataset
- **Training samples:** 1460 houses
- **Target Variable:** `SalePrice`

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Joblib

---

## 📊 Exploratory Data Analysis

Some analyses performed include:

- Distribution of SalePrice
- Correlation Heatmap
- Missing Value Analysis
- Living Area vs SalePrice
- Overall Quality vs SalePrice

Key findings were documented before preprocessing.

## Sample Visualizations

Correlation Heatmap

![Correlation Heatmap](plots/Correlation%20Heatmap.png)

Distribution of SalePrice

![Distribution](plots/Distribution%20of%20Sale%20Price.png)

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

- Removed duplicate rows
- Removed the ID column
- Handled missing values
- Removed extreme outliers
- Applied log transformation to SalePrice
- Ordinal encoding for ordered categorical features
- One-hot encoding for nominal categorical features

---

## 🤖 Models Trained

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

---

## 📈 Model Performance

| Model             |   R² Score |
| ----------------- | ---------: |
| Linear Regression |     0.8963 |
| Decision Tree     |     0.7567 |
| Random Forest     |     0.8762 |
| Gradient Boosting | **0.9063** |
| XGBoost           |     0.8781 |

🏆 **Best Model:** Gradient Boosting Regressor

---

## 📁 Project Structure

```
House-Price-Prediction/
│
├── data/
├── models/
├── plots/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 How to Run

Clone the repository:

```bash
git clone <repository-url>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the preprocessing script:

```bash
python src/data_preprocessing.py
```

Train the models:

```bash
python src/compare_models.py
```

---

## Key Learnings

Through this project, I learned:

- End-to-end machine learning workflow
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Handling missing values and outliers
- Encoding categorical variables
- Comparing multiple regression algorithms
- Model evaluation using MAE, RMSE, and R² Score

---

## 📌 Future Improvements

- Hyperparameter tuning
- Cross-validation
- Feature selection
- Deployment using Flask/FastAPI
- Interactive web application

---

## 👩‍💻 Author

**Jahnvee Sharma**

Machine Learning | Data Science | AI
