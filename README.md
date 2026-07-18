# 🏫 School Data Analysis and Teacher Prediction using Linear Regression

This project demonstrates how to build a **Machine Learning Regression model** using **Linear Regression** to predict the **number of teachers (`jumlah_guru`)** in schools based on other school-related features.

The project uses **Pandas** for data processing and **Scikit-learn** for model training and evaluation.

---

## 📌 Project Overview

The program performs the following tasks:

1. Loads the school dataset (`data.csv`).
2. Cleans the dataset by removing extra spaces from column names.
3. Selects **`jumlah_guru`** as the target variable.
4. Removes unnecessary columns such as **`tahun`**.
5. Identifies numerical and categorical features.
6. Converts categorical variables into numerical values using **One-Hot Encoding**.
7. Splits the dataset into training and testing sets.
8. Trains a **Linear Regression** model.
9. Predicts the number of teachers.
10. Evaluates the model using standard regression metrics.

---

## 📂 Project Structure

```text
.
├── data.csv
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Execute the program using:

```bash
python main.py
```

---

## 📦 Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📊 Machine Learning Model

This project uses the **Linear Regression** algorithm from **Scikit-learn**.

### Workflow

- Load dataset
- Clean column names
- Remove unnecessary columns
- One-Hot Encode categorical variables
- Train/Test Split (80% Training, 20% Testing)
- Train Linear Regression Model
- Predict number of teachers
- Evaluate model performance

---

## 📈 Evaluation Metrics

The model performance is measured using:

- ✅ Mean Absolute Error (MAE)
- ✅ Mean Squared Error (MSE)
- ✅ Root Mean Squared Error (RMSE)
- ✅ R-squared (R² Score)

---

## 🏫 Dataset

The dataset contains school-related information such as:

- School characteristics
- Student information
- School categories
- Regional information
- Number of teachers (`jumlah_guru`)

The target variable for prediction is:

**`jumlah_guru` (Number of Teachers)**

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

---

## 📚 Learning Objectives

This project demonstrates:

- Data preprocessing
- Cleaning column names
- Handling categorical variables
- One-Hot Encoding
- Feature selection
- Linear Regression
- Model evaluation
- Machine Learning workflow using Python

---

## 📄 License

This project is intended for educational and learning purposes.
