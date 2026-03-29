import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ================= LOAD DATA =================
data = pd.read_csv('dataset.csv', header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# ================= SCALING =================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================= MODEL =================
model = SVC(kernel='rbf', C=20, gamma='auto')
model.fit(X_train, y_train)

# ================= PREDICTION =================
y_pred = model.predict(X_test)

# ================= RESULTS =================
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy * 100, "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))