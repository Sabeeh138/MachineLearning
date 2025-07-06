
# 🎓 Student Performance Prediction Web App

This project is a complete machine learning web application that predicts whether a student will **pass or fail** based on their background and performance data. It uses Python for the backend and machine learning, and Flask to serve a simple, clean web interface.

---

## ✅ What Was Done

1. **Data Cleaning and Preprocessing**
   - We used the `StudentsPerformance.csv` dataset.
   - Removed any missing values.
   - Encoded categorical features such as gender, race/ethnicity, parental education, lunch, and test preparation course using `LabelEncoder`.
   - Applied `StandardScaler` to scale all input features for better model performance.

2. **Feature Selection**
   - Used `VarianceThreshold` to remove features with very low variance.

3. **Model Building**
   - Created a new column called `pass` based on the average of math, reading, and writing scores.
   - If a student's average score was **≥ 50**, they were considered to have **passed** (label `1`), otherwise **failed** (label `0`).
   - Used a `RandomForestClassifier` from `sklearn` to train the model on the selected features.

4. **Model Saving**
   - Used `pickle` to save the trained model (`model.pkl`) and the scaler (`scaler.pkl`), so they can be reused in a web app without retraining.

5. **Web Interface**
   - Created a Flask app (`app.py`) that loads the model and scaler.
   - Developed an HTML form in `templates/index.html` where users can input student details.
   - Used CSS (`static/style.css`) to make the interface more visually appealing.
   - Added a background image (`static/background.jpg`) of a university campus to enhance the aesthetic.

---

## 🧠 Technologies Used

| Area            | Tools/Libraries |
|-----------------|-----------------|
| Language        | Python      |
| Data Handling   | pandas, numpy   |
| ML Model        | scikit-learn    |
| Serialization   | pickle          |
| Web Framework   | Flask           |
| Frontend        | HTML, CSS       |
_ _ _
![image](https://github.com/user-attachments/assets/62bfe339-ba04-43cd-b0b8-8b9dd7ea4f6d)
