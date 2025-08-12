Iris Flower Classification

A beginner-friendly **Machine Learning** project that classifies iris flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — based on four measurements: **Sepal Length**, **Sepal Width**, **Petal Length**, and **Petal Width**.

This project is implemented in **Python** using **Jupyter Notebook** and leverages popular ML libraries such as **scikit-learn**, **pandas**, **matplotlib**, and **seaborn**.

---

## 📌 Project Overview

The **goal** is to build and evaluate multiple classification models that can accurately predict the species of an iris flower given its measurements.

### Dataset
- **Source:** [Iris Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- **Features:**
  - SepalLengthCm
  - SepalWidthCm
  - PetalLengthCm
  - PetalWidthCm
- **Target:** Species (*Setosa*, *Versicolor*, *Virginica*)

---

## 🛠 Technologies Used
- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## 🚀 Project Workflow

1. **Import Libraries**  
   Load all required Python packages for data handling, visualization, and ML modeling.

2. **Load Dataset**  
   Read the `Iris.csv` dataset into a pandas DataFrame.

3. **Data Exploration (EDA)**  
   - View dataset head, info, and descriptive statistics.  
   - Check class distribution.
   - Create pairplots and correlation heatmaps.

4. **Data Preprocessing**  
   - Drop unnecessary columns (e.g., `Id` if present).
   - Split into features (X) and target (y).
   - Train-test split (80/20).
   - Standardize numerical features.

5. **Model Training**  
   Train and evaluate the following algorithms:
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
   - Logistic Regression

6. **Evaluation**  
   - Accuracy score
   - Classification report
   - Confusion matrix for the best model

7. **Hyperparameter Tuning**  
   Example grid search for the KNN model to find the best `n_neighbors`.

---

## 📊 Results
- All models achieved high accuracy (> 90%), with **Random Forest** and **SVM** performing best.
- KNN hyperparameter tuning found the optimal `n_neighbors` for improved performance.

---

## 📷 Sample Visualizations

### Pairplot
![Pairplot](screenshots/pairplot.png)

### Correlation Heatmap
![Heatmap](screenshots/heatmap.png)

### Confusion Matrix - Best Model
![Confusion Matrix](screenshots/confusion_matrix.png)

### Model Comparison
![Model Comparison](screenshots/model_comparison.png)

---
## ▶️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/SathwikRThaduru/iris-classification.git
   cd iris-classification
````

2. Install required packages:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open `iris_classification.ipynb` and run cells sequentially.

---

## 📌 Conclusion

This project demonstrates the **complete beginner ML workflow**:

* Data loading
* EDA
* Preprocessing
* Model training
* Evaluation
* Basic hyperparameter tuning

The Iris dataset is small, clean, and well-suited for learning classification techniques, making this project a perfect first step into Machine Learning.
---
✍ Author
Thaduru Sathwik – LinkedIn


