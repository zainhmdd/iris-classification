import torch
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model import MLPClassifier

# Load data
data = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')

X = data.iloc[:, :-1].values
y = pd.Categorical(data.iloc[:, -1]).codes

label_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Load model
model = MLPClassifier()
model.load_state_dict(torch.load('model.pth'))
model.eval()

def predict(model, X):
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)
        return preds.numpy()

# Prediksi contoh
sample_data = [
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.0, 5.2, 2.3]
]
sample_scaled = scaler.transform(sample_data)
predictions = predict(model, sample_scaled)

print("
Prediksi data baru:")
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: Predicted class = {pred} ({label_map[pred]})")

# Evaluasi akurasi MLP
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

with torch.no_grad():
    outputs = model(X_test_t)
    _, y_pred = torch.max(outputs, 1)
acc = (y_pred == y_test_t).sum().item() / len(y_test_t)
print(f"
MLP Test Accuracy: {acc:.4f}")
