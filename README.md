# 🌍 Global Earthquake Magnitude Prediction

This project uses Machine Learning to predict the **earthquake magnitude class** from earthquake-related data.

## 🤖 Model Used

**XGBoost Classifier**

The model was selected after comparing:

* Logistic Regression
* Random Forest
* XGBoost
* Gradient Boosting
* Decision Tree

XGBoost achieved the best overall weighted F1-score.

## 📊 Model Performance

* **F1 Score (Weighted):** 0.823
* **Recall (Weighted):** 0.843
* **Balanced Accuracy:** 0.348

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Joblib

## 🚀 Run the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 📁 Files

* `app.py` – Streamlit application
* `global_earthquake_model.pkl` – Trained XGBoost model
* `requirements.txt` – Required Python libraries
* `README.md` – Project documentation

## 👨‍💻 Author

**Mohammed Farooq Khan**

GitHub: `https://github.com/far00q2241`
