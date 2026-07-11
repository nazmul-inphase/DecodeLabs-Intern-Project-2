import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1. INPUT: Load the dataset
iris = load_iris()
X = iris.data    
y = iris.target  

# 2. PROCESS: Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. PROCESS: Train KNN Engine
model = KNeighborsClassifier(n_neighbors=11)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

# 4. OUTPUT: Text Diagnostics
print("\n================ DIAGNOSTIC REPORT ================")
print(classification_report(y_test, predictions, target_names=iris.target_names))
print("===================================================")

# 5. VISUAL OUTPUT: Beautiful Confusion Matrix Plot
print("Generating visual confusion matrix chart...")
cm = confusion_matrix(y_test, predictions)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

# Plot with a nice color map (blues)
display.plot(cmap=plt.cm.Blues)
plt.title("Iris Classification - Confusion Matrix")
plt.show()