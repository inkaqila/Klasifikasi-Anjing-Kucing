import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def extract_features(image_path):
    try:
        image = Image.open(image_path).convert('RGB').resize((64, 64))
        return np.array(image).flatten()
    except:
        return None

X, y = [], []
for folder_name, label in [('cats', 'cat'), ('dogs', 'dog')]:
    folder = os.path.join('dataset', folder_name)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        features = extract_features(path)
        if features is not None:
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')
print("✅ Model saved as model.pkl")

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred, labels=['cat', 'dog'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['cat', 'dog'], yticklabels=['cat', 'dog'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('static/confusion_matrix.png')
print("🖼️ Confusion matrix saved to static/confusion_matrix.png")
