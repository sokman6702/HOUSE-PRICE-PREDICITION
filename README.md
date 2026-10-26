# 🏡 House Price Predictor

A machine learning web application that estimates house prices based on property characteristics using **Linear Regression**, Python, and Streamlit.

The project demonstrates an end-to-end beginner machine learning workflow, including data preparation, model training, evaluation, and interactive prediction through a web dashboard.

## 📌 Project Overview

The House Price Predictor allows users to enter details about a house and receive an estimated price in Pakistani Rupees (PKR).

The application also provides model performance metrics, actual-versus-predicted price comparisons, dataset exploration, and feature correlation analysis.

**Note:** This project uses a small, illustrative dataset for educational purposes. Its predictions are not suitable for real estate transactions or reliable market valuation.

## ✨ Features

- 🏠 **Interactive Price Prediction** — Enter house details to estimate its price.
- 📊 **Model Performance Dashboard** — View MAE, RMSE, and R² score.
- 📈 **Actual vs. Predicted Comparison** — Compare model predictions with actual test-set prices.
- 🔍 **Data Insights** — Explore the dataset, price distribution, and feature correlations.
- 🧭 **Streamlit Interface** — Navigate the application through a clean, tab-based dashboard.
- 💰 **PKR Formatting** — Display estimated prices in Pakistani Rupees.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data handling and analysis |
| NumPy | Numerical calculations |
| Scikit-learn | Model training, data splitting, and evaluation |
| Streamlit | Interactive web application |
| Matplotlib | Data visualization support |

## 🏗️ Machine Learning Workflow

1. Create and explore the house-price dataset.
2. Separate the input features from the target variable.
3. Split the dataset into training and testing sets.
4. Train a Linear Regression model.
5. Generate predictions for the test data.
6. Evaluate the model using regression metrics.
7. Build an interactive dashboard for predictions and data exploration.

## 📂 Input Features

The model uses the following features:

| Feature | Description |
|---|---|
| `Area_sqft` | House area in square feet |
| `Bedrooms` | Number of bedrooms |
| `Bathrooms` | Number of bathrooms |
| `Age_years` | Age of the house in years |

**Target variable:** `Price_PKR` — the house price in Pakistani Rupees.

## 📊 Model Evaluation

The current project has produced the following evaluation results:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | ₨653,335 |
| Root Mean Squared Error (RMSE) | ₨665,207 |
| R² Score | 0.803 |

These results are from the current train-test split. The test set contains only two houses, so the metrics are highly sensitive to individual examples and do not establish real-world predictive performance. An R² score of 0.803 should not be interpreted as 80.3% prediction accuracy.

## 🚀 Getting Started

### Prerequisites

- Python installed on your computer
- Visual Studio Code or another code editor
- Basic knowledge of Python

### 1. Clone the repository

After publishing this project on GitHub, run:

```bash
git clone https://github.com/YOUR-USERNAME/House-Price-Predictor.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Open the project folder

```bash
cd House-Price-Predictor
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually:

`http://localhost:8501`

Open the URL in your browser to use the dashboard.

## 📁 Project Structure

```text
House-Price-Predictor/
├── app.py
├── README.md
├── requirements.txt
└── screenshots/
    ├── dashboard.png
    ├── prediction.png
    └── model-performance.png
```

The screenshots folder will contain application screenshots once they have been captured.

## 🖥️ Application Screenshots

Add screenshots of your running application to the `screenshots/` folder.

### Dashboard

![House Price Predictor Dashboard](screenshots/dashboard.png)

### Price Prediction

![House Price Prediction](screenshots/prediction.png)

### Model Performance

![Model Performance](screenshots/model-performance.png)

## 🎯 Learning Objectives

This project was developed to practice:

- Working with structured data using Pandas.
- Separating features and target variables.
- Training a supervised machine learning regression model.
- Splitting data into training and testing sets.
- Evaluating predictions with regression metrics.
- Building a machine learning interface using Streamlit.
- Organizing a project for GitHub and portfolio presentation.

## 🔮 Future Improvements

- Use a larger, real-world house-price dataset.
- Improve model evaluation with cross-validation.
- Compare Linear Regression with other regression algorithms.
- Add data validation and better error handling.
- Improve visualizations and user experience.
- Deploy the application online.

## 👨‍💻 Author

**AI/ML Student | Aspiring AI Engineer**

This project is part of my journey toward becoming an AI Engineer through practical machine learning projects.

## 📄 License

This project is available for educational and learning purposes. Add an explicit open-source license if you intend to grant others formal permission to use, modify, and distribute the code.
