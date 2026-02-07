# Diabetes Prediction Using Machine Learning and Neural Networks

## 📌 Project Overview

This project focuses on predicting the likelihood of diabetes in patients using supervised machine learning and deep learning techniques. Two models are implemented and compared:

* **XGBoost Classifier**
* **Artificial Neural Network (ANN)**

The objective is to analyze medical data, build predictive models, and evaluate their performance using standard classification metrics.

---

## 🧠 Methodology

The project follows a systematic machine learning workflow:

1. **Data Collection**
   A structured medical dataset containing patient health attributes is used.

2. **Data Preprocessing**

   * Checked for missing values
   * Separated features and target variable
   * Applied feature scaling using StandardScaler

3. **Train–Test Split**

   * Dataset split into 80% training and 20% testing data

4. **Model Building**

   * XGBoost Classifier for ensemble-based learning
   * Artificial Neural Network with fully connected layers

5. **Model Training**

   * Models trained on scaled training data

6. **Evaluation**

   * Predictions made on test data
   * Performance evaluated using accuracy, precision, recall, and F1-score

---

## 📥 Input

The input consists of medical attributes of patients:

* Pregnancies
* Glucose Level
* Blood Pressure
* Skin Thickness
* Insulin Level
* Body Mass Index (BMI)
* Diabetes Pedigree Function
* Age

Each input record represents one patient.

---

## 📤 Output

* **Binary Classification Result**:

  * `0` → Non-diabetic
  * `1` → Diabetic

The model predicts whether a patient is likely to have diabetes based on the input features.

---

## ⚙️ Models Used

### 1️⃣ XGBoost Classifier

* Gradient boosting-based ensemble model
* Handles non-linear relationships efficiently
* Performs well on structured/tabular data

### 2️⃣ Artificial Neural Network (ANN)

* Multi-layer perceptron architecture
* Uses ReLU activation in hidden layers
* Sigmoid activation for binary output

---

## 📈 Performance & Accuracy

| Model                     | Accuracy |
| ------------------------- | -------- |
| XGBoost Classifier        | ~75–80%  |
| Artificial Neural Network | ~73–78%  |

> *Exact accuracy may vary depending on training conditions and random state.*

Additional evaluation metrics used:

* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 🛠 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib & Seaborn
* Scikit-learn
* XGBoost
* TensorFlow / Keras
* Jupyter Notebook

---

## ▶️ How to Run the Project

1. Install required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow
```

2. Open the notebook:

```bash
jupyter notebook
```

3. Run all cells sequentially to train and evaluate the models.

---

## 📌 Conclusion

This project demonstrates how machine learning and neural networks can be effectively applied to healthcare data for early disease prediction. It highlights the importance of data preprocessing, model selection, and evaluation in building reliable predictive systems.
