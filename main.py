from sklearn.datasets import load_iris
import pandas as pd

from src.data_preprocessing import load_data, split_data, scale_features
from src.model_training import train_knn, save_model
from src.model_evaluation import evaluate_model, plot_confusion_matrix
from src.prediction import predict


# 1. Load dataset
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# 2. Preprocessing
X, y = load_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)
X_train, X_test, scaler = scale_features(X_train, X_test)

# 3. Train model
model = train_knn(X_train, y_train)

# 4. Save model
save_model(model)

# 5. Evaluate model
y_pred = evaluate_model(model, X_test, y_test)
plot_confusion_matrix(y_test, y_pred)

# 6. Test prediction
sample = [5.1, 3.5, 1.4, 0.2]
result = predict(model, scaler, sample)

print("Prediction:", result)