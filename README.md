# DecodeLabs-Intern-Project-2
Data Classification Using AI 

# Iris Flower Classification using KNN

An end-to-end Machine Learning pipeline to classify Iris flower species using the K-Nearest Neighbors (KNN) algorithm.

## Project Methodology
- **Input:** Loaded the famous Iris dataset (150 samples, 4 features).
- **Process:** Performed 80/20 train-test split, scaled features using `StandardScaler`, and optimized with `K-Neighbors Classifier (K=11)`.
- **Output:** Achieved 100% accuracy, evaluated via a diagonal Confusion Matrix and Classification Report (F1-Score).

## Key Skills Demonstrated
- Supervised Machine Learning & Data Preprocessing
- Model Evaluation & Performance Diagnostics
- Data Visualization with Matplotlib

---

## Performance Metrics & Evaluation

### 1. Confusion Matrix
A tabular layout that allows visualization of the performance of the algorithm. Our matrix achieved a perfect **diagonal form**, indicating zero misclassifications (0% Error Rate) across all three Iris species (Setosa, Versicolor, and Virginica).

### 2. Precision & Recall
- **Precision (1.00):** Out of all instances the model predicted as a specific flower, 100% of them were correct. There are no False Positives.
- **Recall (1.00):** Out of all actual flowers present in the test set, the model successfully identified 100% of them. There are no False Negatives.

### 3. F1-Score (1.00)
The harmonic mean of Precision and Recall. A perfect score of 1.00 signifies that the model is perfectly optimized and robust, maintaining an ideal balance without any classification bias.

![Confusion Matrix](Confusion%20Matrix.png)